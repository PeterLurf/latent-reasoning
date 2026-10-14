import math
d = [("s182_window_s2",0.001,0.470),("s184_reweight_s2",0.003,0.436),("s185_rms_s2",0.003,0.588),("s185_rms_s1",0.006,0.625),("s190_rmsnopass_s1",0.002,0.480),
     ("s187_nopass_s1",0.014,0.472),("s187_nopass_s2",0.005,0.474),("s184_reweight_s1",0.110,0.473),("s190_rmsnopass_s2",0.228,0.481),("s182_window_s1",0.239,0.529),
     ("s182_nofade_s1",0.560,0.768),("s185_den_s1",0.642,0.930),("s188_rwwin_s1",0.901,0.763),("s182_hold_s1",1.060,0.898),("s185_rms_s3",1.031,0.978)]
for r, loss, c in d:
    print("%-20s loss %.3f  delta_cos %.3f  lambda_eff = ln(2(1-cos))/32 = %+.3f" % (r, loss, c, math.log(2 * (1 - c)) / 32))
