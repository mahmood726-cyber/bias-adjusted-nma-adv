import numpy as np
from bias_nma_adv.survival import SurvivalIPDReconstructor

def test_guyot_ipd_reconstruction():
    reconstructor = SurvivalIPDReconstructor()
    
    # Mock KM curve: starts at 1.0 survival at t=0, drops to 0.8 at t=5, 0.5 at t=10
    times = [0.0, 2.5, 5.0, 7.5, 10.0]
    survivals = [1.0, 0.9, 0.8, 0.6, 0.5]
    
    # At-risk table: 100 patients at start, 80 at t=5, 50 at t=10
    n_risk_times = [0.0, 5.0, 10.0]
    n_risk_values = [100, 80, 50]
    
    reconstructor.add_arm_curve(
        arm_id=1,
        times=times,
        survivals=survivals,
        n_risk_times=n_risk_times,
        n_risk_values=n_risk_values,
        total_n=100
    )
    
    t_arr, e_arr, arm_arr = reconstructor.get_combined_ipd()
    
    # Total patient count should equal total_n (100)
    assert len(t_arr) == 100
    assert len(e_arr) == 100
    assert len(arm_arr) == 100
    
    # All arms should be labeled as arm=1
    assert np.all(arm_arr == 1)
    
    # There should be both events (1) and censored observations (0)
    assert np.any(e_arr == 1)
    assert np.any(e_arr == 0)
    
    # Reconstructed times should all fall within the KM range [0.0, 10.0]
    assert np.all((t_arr >= 0.0) & (t_arr <= 10.0))
