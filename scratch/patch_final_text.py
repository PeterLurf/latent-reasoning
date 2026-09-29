import pathlib
p = pathlib.Path(r"D:\Latent\report\render3.js")
s = p.read_text(encoding="utf-8")
rep = [
 ('vRow({s3: "C", s4: "D", s5: "I"}[s],', 'vRow({s3: "C", s4: "D", s5: "I1"}[s],'),
 ('vRow("I", "Second K = 4 seed:', 'vRow("I2", "Second K = 4 seed:'),
 ("Two further seeds, trained after Part II's predictions were frozen, test whether those predictions hold on seeds that played no part in forming them.",
  "Three further seeds, trained after the predictions they are scored against were frozen, test whether Part II's predictions hold on seeds that played no part in forming them."),
 ("Training on this rig proved fragile: a host crash killed the first runs, and several bf16 runs stalled at the first curriculum stage and were restarted under a rule written before their evaluation (<a href=\"#integrity\">details</a>).",
  "Training on this rig proved fragile. Two host crashes killed runs in progress, one GPU dropped off the bus at the end, and about half of all gated training attempts stalled at the first curriculum stage; those were restarted under a rule written before their evaluation (<a href=\"#integrity\">details</a>). Several of the new runs still failed the competence gate and are reported as training failures."),
]
for a, b in rep:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
a = "<p>Training on this rig is not deterministic even with a fixed seed and device,"
b = "<p>A second crash followed at about 07:20 (bugcheck 0x133, a DPC watchdog timeout), and at the end of the session the RTX 3070 dropped off the PCI bus (\"GPU is lost\"), taking the last evaluation job with it; that model had already failed the competence gate. Runs launched after the second crash keep a full training state and resume after a crash.</p>\n  <p>Training on this rig is not deterministic even with a fixed seed and device,"
assert s.count(a) == 1
s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("ok")
