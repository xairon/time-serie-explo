"""Assign all ownerless models (MLflow runs in the Junon_TimeSeries and pastas
experiments) and datasets (config.yaml under data/prepared) to a given admin email.

Usage (run inside the backend container after deploy):
    python scripts/assign_legacy_ownership.py --email admin@x.fr
"""
import argparse
import asyncio
from pathlib import Path

import yaml
from sqlalchemy import select

from api.config import settings
from api.database import async_session
from api.models_db import User

EXPERIMENTS = ["Junon_TimeSeries", "pastas"]


async def _resolve_owner(email: str) -> str:
    async with async_session() as db:
        user = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
        if user is None:
            raise SystemExit(f"No user with email {email}")
        return str(user.id)


def _tag_models(owner_id: str) -> int:
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    n = 0
    for exp_name in EXPERIMENTS:
        experiment = client.get_experiment_by_name(exp_name)
        if experiment is None:
            continue
        runs = client.search_runs(experiment_ids=[experiment.experiment_id], filter_string="")
        for run in runs:
            if "owner_id" not in run.data.tags:
                client.set_tag(run.info.run_id, "owner_id", owner_id)
                n += 1
    return n


def _tag_datasets(owner_id: str) -> int:
    prepared = Path(settings.data_dir) / "prepared"
    n = 0
    if not prepared.exists():
        return 0
    for cfg in prepared.rglob("config.yaml"):
        data = yaml.safe_load(cfg.read_text()) or {}
        if not data.get("owner_id"):
            data["owner_id"] = owner_id
            cfg.write_text(yaml.safe_dump(data, default_flow_style=False, allow_unicode=True))
            n += 1
    return n


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--email", required=True)
    args = p.parse_args()
    owner_id = await _resolve_owner(args.email)
    print(f"Models tagged: {_tag_models(owner_id)}")
    print(f"Datasets tagged: {_tag_datasets(owner_id)}")


if __name__ == "__main__":
    asyncio.run(main())
