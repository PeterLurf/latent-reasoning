import numpy as np
R=np.random.default_rng(27499); N=40000
clip=lambda x:np.clip(x,0,1)
print("R-WP-DEPTH (design): P(pass) by shift, sig .09")
for dl in np.arange(0,-0.161,-0.02):
    z=R.standard_normal((N,3)); a6=R.binomial(500,clip(0.965+dl+.09*z))/500; a7=R.binomial(500,clip(0.900+dl+.09*z))/500
    print(" %+.2f %.3f"%(dl,((a6.mean(1)>=0.921)&(a7.mean(1)>=0.859)).mean()))
print("R-SO-DEPTH (design): P(pass) by delta, sig .09")
for dl in np.arange(0,-0.241,-0.02):
    js=clip(.85+dl+.09*R.standard_normal((N,3))); ss=clip(.85+.09*R.standard_normal((N,3)))
    d=(R.binomial(1060,js)/1060).mean(1)-(R.binomial(1060,ss)/1060).mean(1); print(" %+.2f %.3f"%(dl,(d>=-0.10).mean()))
print("Alt: WP joint vs ctl_wp x3 (unpaired, d7 only, bar diff>=-0.05), ctl drift +0.03 / joint shift dl relative to ctl")
for dl in (0,-0.05,-0.10,-0.15):
    c=clip(.93+.09*R.standard_normal((N,3))); j=clip(.93+dl+.09*R.standard_normal((N,3)))
    d=(R.binomial(500,j)/500).mean(1)-(R.binomial(500,c)/500).mean(1); print(" %+.2f P(pass) %.3f"%(dl,(d>=-0.05).mean()))
print("Masking: true ctl drift +0.05 (continued WP training improves d6/d7), joint = of_s1 level (i.e. -0.05 interference vs ctl)")
z=R.standard_normal((N,3)); a6=R.binomial(500,clip(0.965+.09*z))/500; a7=R.binomial(500,clip(0.900+.09*z))/500
print(" design R-WP-DEPTH pass (=> NO-INTERFERENCE) %.3f"%((a6.mean(1)>=0.921)&(a7.mean(1)>=0.859)).mean())
print("Momentum share: beta1 .9 strict alternation, own-family weight in m_t:", sum(0.1*0.9**k for k in range(0,400,2))/sum(0.1*0.9**k for k in range(400)))
print("Adam v-hat: if |g_WP|<<|g_SO|, SO eff. step ratio joint/single ~ sqrt(2) =",2**.5)
