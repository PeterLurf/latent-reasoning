import sys, numpy as np
sys.path.insert(0, "D:/Latent")
import boxes.ow_words as OW, boxes.inject_ow as IO
ENC = IO.ENC; D = 768; V = 50257
blk = lambda T: 24 * D * D * T + 2 * D * T * (T + 1)
head = 2 * D * V
def step_tok(c, ctx):           # c new tokens through 12 blocks attending to ctx cached tokens
    return 12 * (24 * D * D * c + 4 * D * c * (ctx + c))
s = OW.Stream(IO.SEED_TEST2_REG, frames=OW.FRAME_TRAIN, **IO.OWKW, mul_s_p=0)
for d in (2, 3, 4, 6, 8):
    its = [s.draw(d) for _ in range(200)]
    P = np.mean([len(ENC.encode(it["text"])) for it in its])
    stepT = np.mean([np.mean([len(ENC.encode(" " + r)) for r in it["R_step"]]) for it in its])
    ansT = len(ENC.encode(" The answer is 10."))
    # TEXT: prefill P, then d steps of stepT tokens + answer tokens, KV cached, head per generated token
    n_gen = d * stepT + ansT
    F_text = 12 * blk(P) + sum(step_tok(1, P + i) + head for i in range(int(round(n_gen))))
    # CC: prefill P once (no memory), then K'=d-1 passes of cue c tokens each + decoder c_dec tokens; head once per pass + decoder; memory r/w ~ 2*8*D*D per cue token (q,k,v,o proj)
    def F_cc(c, c_dec, Kp):
        per = step_tok(c, P) + 8 * D * D * c + head
        return 12 * blk(P) + Kp * per + step_tok(c_dec, P) + 8 * D * D * c_dec + head
    long_c = len(ENC.encode(" How many eggs does Kim have? The answer is:"))
    print("d", d, "P %.0f stepT %.1f" % (P, stepT), "TEXT %.1f G" % (F_text / 1e9),
          "CC long-cue(%d) %.1f G" % (long_c, F_cc(long_c, long_c, d - 1) / 1e9),
          "CC short-cue(2) %.1f G" % (F_cc(2, 4, d - 1) / 1e9),
          "prefill %.1f G" % (12 * blk(P) / 1e9))
