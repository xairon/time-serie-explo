"""Compatibility layer for TimeSHAP with modern SHAP versions.

Fixes the import error: cannot import name 'Kernel' from 'shap.explainers._kernel'
by patching the module before TimeSHAP imports it.
"""

import sys

def patch_shap_for_timeshap():
    """
    Patch SHAP module to be compatible with TimeSHAP.
    
    SHAP >= 0.45 renamed 'Kernel' to 'KernelExplainer' in the internal module,
    but TimeSHAP still imports from the old name.
    """
    try:
        import shap.explainers._kernel as kernel_module
        
        # Check if patch is needed
        if not hasattr(kernel_module, 'Kernel') and hasattr(kernel_module, 'KernelExplainer'):
            # Create alias
            kernel_module.Kernel = kernel_module.KernelExplainer
            print("[timeshap_compat] Patched SHAP Kernel alias")
            return True
        elif hasattr(kernel_module, 'Kernel'):
            # Already has Kernel, no patch needed
            return True
        else:
            print("[timeshap_compat] Warning: Could not find Kernel or KernelExplainer")
            return False
            
    except ImportError as e:
        print(f"[timeshap_compat] Could not import shap._kernel: {e}")
        return False


def import_timeshap():
    """
    Import TimeSHAP after applying compatibility patches.
    Returns the timeshap module or None if import fails.
    """
    # Apply patch first
    patch_shap_for_timeshap()
    
    try:
        import timeshap
        from timeshap.explainer import local_event, local_feat
        from timeshap.plot import plot_event_heatmap, plot_feat_barplot
        
        return {
            'timeshap': timeshap,
            'local_event': local_event,
            'local_feat': local_feat,
            'plot_event_heatmap': plot_event_heatmap,
            'plot_feat_barplot': plot_feat_barplot
        }
    except ImportError as e:
        print(f"[timeshap_compat] Failed to import timeshap: {e}")
        return None


# Auto-patch on import
_patched = patch_shap_for_timeshap()
