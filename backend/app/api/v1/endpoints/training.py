"""
Training endpoints - Start and monitor training jobs.

Uses background tasks for training execution.
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException
import asyncio
import logging
from datetime import datetime
from uuid import uuid4

from app.schemas import (
    TrainingRequest,
    TrainingJobStatus,
    TrainingJobResponse,
)
from app.config import get_settings

import sys
from pathlib import Path

# Add legacy dashboard to path if needed (for legacy pickle compat)
sys.path.insert(0, str(Path(__file__).parents[5]))

router = APIRouter()

# In-memory job tracking (would be Redis in production)
_jobs: dict[str, TrainingJobStatus] = {}

# Logger
logger = logging.getLogger(__name__)


@router.post("/start", response_model=TrainingJobResponse)
async def start_training(request: TrainingRequest):
    """
    Start a new training job.
    
    Returns immediately with a job_id that can be used to monitor progress.
    """
    try:
        job_id = str(uuid4())
        
        # Create job status
        _jobs[job_id] = TrainingJobStatus(
            job_id=job_id,
            status="pending",
            progress=0.0,
            total_epochs=request.config.epochs,
            started_at=datetime.now(),
        )
        
        # Launch background training task
        asyncio.create_task(_run_training_background(job_id, request))
        
        return TrainingJobResponse(
            job_id=job_id,
            status="pending",
            message=f"Training job started. Stations: {request.stations}, Model: {request.config.model_name}",
        )
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_training_background(job_id: str, request: TrainingRequest):
    """
    Background wrapper to run the synchronous training logic in an executor.
    """
    try:
        _jobs[job_id].status = "running"
        
        loop = asyncio.get_running_loop()
        
        # Run the CPU-bound training logic in a separate thread
        results = await loop.run_in_executor(
            None,
            execute_training_job,
            job_id,
            request
        )
        
        if results.get('status') == 'success':
            _jobs[job_id].status = "completed"
            _jobs[job_id].progress = 100.0
            _jobs[job_id].completed_at = datetime.now()
            # Extract basic model usage info
            saved_path = results.get('saved_path', '')
            model_id = Path(saved_path).name if saved_path else f"{request.config.model_name}_{job_id[:8]}"
            _jobs[job_id].model_id = model_id
        else:
            _jobs[job_id].status = "failed"
            _jobs[job_id].error = results.get('error', 'Unknown error')
            
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")
        _jobs[job_id].status = "failed"
        _jobs[job_id].error = str(e)


def execute_training_job(job_id: str, request: TrainingRequest) -> Dict[str, Any]:
    """
    Synchronous function executing the full training pipeline.
    This runs in a worker thread.
    """
    try:
        # Import core modules here to avoid import loops or startup overhead
        from core.dataset_registry import get_dataset_registry
        from core.preprocessing import (
            TimeSeriesPreprocessor,
            prepare_dataframe_for_darts,
            split_train_val_test,
            add_datetime_features,
            add_lag_features
        )
        from core.training import run_training_pipeline
        
        settings = get_settings()
        
        # 1. Load Dataset
        registry = get_dataset_registry()
        # Find dataset by name/id
        datasets = registry.scan_datasets()
        dataset_info = next((ds for ds in datasets if ds.name == request.dataset_id), None)
        
        if not dataset_info:
            return {'status': 'error', 'error': f"Dataset '{request.dataset_id}' not found"}
        
        df, dataset_config = registry.load_dataset(dataset_info)
        
        # 2. Filter by stations if requested (and available)
        # Note: DatasetRegistry stores multi-station in single DF usually with a station column
        # BUT current prepare_dataframe_for_darts assumes single series logic or simple pivoting.
        # For simplicity in Phase 2b:
        # - If 1 station requested: filter DF
        # - If >1 stations: use global model logic (requires adaptation)
        
        # Check if dataset has station column
        station_col = dataset_config.get('station_column')
        target_col = dataset_config.get('target_column')
        covariate_cols = dataset_config.get('covariate_columns', [])
        
        requested_stations = request.stations if request.stations else dataset_config.get('stations', [])
        
        # PREPARATION FOR DARTS
        # We need to create TimeSeries for each station
        
        train_list = []
        val_list = []
        test_list = []
        
        train_cov_list = []
        val_cov_list = []
        test_cov_list = []
        
        full_cov_list = []
        
        # Scalers storage
        target_preprocessors = {}
        cov_preprocessors = {}
        
        # Handle single vs multi station
        processing_stations = requested_stations
        
        # Ensure we have data for these stations
        if station_col:
            df = df[df[station_col].isin(processing_stations)]
        
        # Helper to update progress - tricky from thread, so we update the shared dict directly
        # (This is thread-safe enough for simple dict assignment in CPython due to GIL)
        def update_progress(p):
            if job_id in _jobs:
                _jobs[job_id].progress = p
        
        # 3. Process each station
        total_stations = len(processing_stations)
        
        for i, station in enumerate(processing_stations):
            update_progress(10 + (i / total_stations * 20)) # 10-30% progress
            
            # Filter data for station
            if station_col:
                station_df = df[df[station_col] == station].copy()
            else:
                station_df = df.copy() # Assume single station dataset
            
            if station_df.empty:
                logger.warning(f"No data for station {station}")
                continue
                
            # Create TimeSeries
            ts_target, ts_cov = prepare_dataframe_for_darts(
                station_df,
                target_col=target_col,
                covariate_cols=covariate_cols,
                fill_method=dataset_config.get('preprocessing', {}).get('fill_method', 'Supprimer les lignes')
            )
            
            # Split
            # Use request config or defaults
            # Assuming request.config might have split ratios, but using defaults for now
            train, val, test = split_train_val_test(
                ts_target,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15
            )
            
            # Split covariates if they exist
            if ts_cov:
                train_c, val_c, test_c = split_train_val_test(ts_cov, 0.7, 0.15, 0.15)
            else:
                train_c, val_c, test_c = None, None, None
            
            # Preprocessing (Scaling)
            # Create a PREPROCESSOR for this station
            # Note: We create one preprocessor per station usually, OR one global.
            # Here we follow "local scaling" usually preferred for hydrology unless we specifically want global.
            # Given preprocessor design:
            
            preproc_config = {
                'fill_method': dataset_config.get('preprocessing', {}).get('fill_method', 'Supprimer les lignes'),
                'normalization': request.config.hyperparams.get('scaler_type', 'StandardScaler (z-score)'),
                'transformation': request.config.hyperparams.get('transformation', 'Aucune'),
            }
            
            preproc = TimeSeriesPreprocessor(preproc_config)
            train_scaled = preproc.fit_transform(train)
            val_scaled = preproc.transform(val)
            test_scaled = preproc.transform(test)
            
            target_preprocessors[station] = preproc
            
            # Covariates preprocessing
            cov_preproc = None
            if train_c:
                cov_preproc = TimeSeriesPreprocessor(preproc_config)
                train_c_scaled = cov_preproc.fit_transform(train_c)
                val_c_scaled = cov_preproc.transform(val_c)
                test_c_scaled = cov_preproc.transform(test_c)
                cov_preprocessors[station] = cov_preproc
            else:
                train_c_scaled, val_c_scaled, test_c_scaled = None, None, None

            # Add datetime features (Future Covariates) if requested
            # Usually added to covariates list
            if request.config.use_covariates: # Assuming simple boolean
                # Generate datetime features
                # Note: datetime features need to be generated for the FULL range ideally
                # Here we generate for each split
                
                # Helper to add features
                def add_feats(series_target):
                    return add_datetime_features(series_target)

                # Darts models handle future covariates differently.
                # Simplest path: Add datetime features to PAST covariates if model supports past,
                # or create FUTURE covariates.
                # run_training_pipeline expects `train_cov`, `val_cov` which are passed as past_covariates.
                # We will append datetime features to these covariates.
                
                dt_train = add_feats(train)
                dt_val = add_feats(val)
                dt_test = add_feats(test)
                
                if train_c_scaled:
                    train_c_scaled = train_c_scaled.stack(dt_train)
                    val_c_scaled = val_c_scaled.stack(dt_val)
                    test_c_scaled = test_c_scaled.stack(dt_test)
                else:
                    train_c_scaled = dt_train
                    val_c_scaled = dt_val
                    test_c_scaled = dt_test

            train_list.append(train_scaled)
            val_list.append(val_scaled)
            test_list.append(test_scaled)
            
            train_cov_list.append(train_c_scaled)
            val_cov_list.append(val_c_scaled)
            test_cov_list.append(test_c_scaled)
            
            # Full cov for prediction later ?
            # Not strictly needed for training pipeline as it uses splits
            
        update_progress(40.0)
        
        # 4. Configure Training
        
        # Handle Single vs Global
        # If 1 station -> pass single TimeSeries
        # If multiple -> pass List[TimeSeries]
        
        if len(train_list) == 1:
            train_input = train_list[0]
            val_input = val_list[0]
            test_input = test_list[0]
            train_cov_input = train_cov_list[0]
            val_cov_input = val_cov_list[0]
            test_cov_input = test_cov_list[0]
            # Use single preprocessor
            target_preproc_input = list(target_preprocessors.values())[0]
            cov_preproc_input = list(cov_preprocessors.values())[0] if cov_preprocessors else None
            # Station name
            station_name_input = processing_stations[0]
            all_stations_input = None
        else:
            train_input = train_list
            val_input = val_list
            test_input = test_list
            train_cov_input = train_cov_list
            val_cov_input = val_cov_list
            test_cov_input = test_cov_list
            target_preproc_input = target_preprocessors
            cov_preproc_input = cov_preprocessors
            # Global name
            station_name_input = f"Global_{len(processing_stations)}_stations"
            all_stations_input = processing_stations
            
        # Parse hyperparams from request
        hyperparams = request.config.hyperparams.copy()
        hyperparams['model_name'] = request.config.model_name
        hyperparams['epochs'] = request.config.epochs
        hyperparams['batch_size'] = request.config.batch_size
        
        # PL Trainer kwargs
        pl_trainer_kwargs = {
            'accelerator': 'auto',
            'callbacks': [] # Could add custom callback for progress
        }
        
        # 5. Run Pipeline
        # We need raw DF for saving. We can use the original 'df' if single station, or reconstruct.
        # run_training_pipeline expects `station_data_df` which is the FULL DataFrame.
        
        update_progress(50.0)
        
        pipeline_results = run_training_pipeline(
            model_name=request.config.model_name,
            hyperparams=hyperparams,
            train=train_input,
            val=val_input,
            test=test_input,
            train_cov=train_cov_input,
            val_cov=val_cov_input,
            test_cov=test_cov_input,
            use_covariates=request.config.use_covariates,
            save_dir=settings.checkpoints_dir / "darts",
            station_name=station_name_input,
            verbose=True,
            pl_trainer_kwargs=pl_trainer_kwargs,
            station_data_df=df, # Pass full DF, logic handles splitting for save
            station_data_df_raw=None, # Will be reconstructed
            target_preprocessor=target_preproc_input,
            cov_preprocessor=cov_preproc_input,
            original_filename=dataset_info.source_file,
            preprocessing_config=dataset_config.get('preprocessing'),
            all_stations=all_stations_input,
            early_stopping_patience=request.config.early_stopping_patience
        )
        
        return pipeline_results
        
    except Exception as e:
        import traceback
        logger.error(f"Error in training execution: {e}")
        logger.error(traceback.format_exc())
        return {'status': 'error', 'error': str(e)}


@router.get("/{job_id}/status", response_model=TrainingJobStatus)
async def get_training_status(job_id: str):
    """
    Get the current status of a training job.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    return _jobs[job_id]


@router.delete("/{job_id}")
async def cancel_training(job_id: str):
    """
    Cancel a running training job.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = _jobs[job_id]
    
    if job.status not in ["pending", "running"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job in status '{job.status}'",
        )
    
    # In a real implementation with Celery, we would revoke the task.
    # With asyncio.run_in_executor, cancelling is hard.
    # For Phase 2b: we just mark it cancelled in DB. The thread will continue but result ignored.
    job.status = "cancelled"
    
    return {"message": f"Job '{job_id}' cancelled"}


@router.get("")
async def list_training_jobs(
    status: Optional[str] = None,
    limit: int = 50,
):
    """
    List recent training jobs.
    """
    jobs = list(_jobs.values())
    
    if status:
        jobs = [j for j in jobs if j.status == status]
    
    # Sort by started_at descending
    jobs.sort(key=lambda j: j.started_at or "", reverse=True)
    
    return {"jobs": jobs[:limit], "total": len(jobs)}
