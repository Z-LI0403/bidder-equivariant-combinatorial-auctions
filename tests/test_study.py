import json
import unittest
import torch
from src.train import ROOT
from src.models import build_model
from src.analyze_study import t4_critical,contrast


class StudyTests(unittest.TestCase):
    def test_frozen_matrix_and_parameter_counts(self):
        manifest=json.loads((ROOT/"configs/study_v1/manifest.json").read_text())
        self.assertEqual(manifest["unique_training_runs"],205)
        self.assertEqual(len(manifest["jobs"]),205)
        for key,path in manifest["jobs"].items():
            config=json.loads((ROOT/path).read_text())
            self.assertEqual(config["experiment_id"],key)
            self.assertEqual(config["seeds"],[11,22,33,44,55])
            self.assertEqual(config["report_upper"],3*config["m"]+config["m"]*(config["m"]-1)//2)
            if config["seed"] == 11:
                model=build_model(config)
                self.assertEqual(config["parameter_count"],sum(p.numel() for p in model.parameters()))
            self.assertLess(abs(config.get("parameter_count_relative_difference",0)),.006)

    def test_paired_statistics(self):
        self.assertAlmostEqual(t4_critical(),2.7764451051977987,places=10)
        result=contrast([1,2,3,4,5],[0,1,2,3,4])
        self.assertEqual(result["mean"],1.)
        self.assertEqual(result["std"],0.)
        self.assertEqual(result["ci95"],[1.,1.])

    def test_final_seed_pairing_on_saved_data(self):
        progress=ROOT/"results/study_v1/progress.json"
        if not progress.exists(): self.skipTest("No completed final jobs yet")
        records=json.loads(progress.read_text())["completed"]
        selected=[ROOT/r["run"] for key,r in records.items() if key.startswith("n2_m3_N20000_b0p5_d0p0_") and key.endswith("_s11")]
        if len(selected)<2: self.skipTest("Need two completed paired models")
        if not all((directory/"datasets.pt").exists() for directory in selected):
            self.skipTest("Optional raw-tensor audit: regenerate tensors with src.reproduce")
        first=torch.load(selected[0]/"datasets.pt",weights_only=True)
        for directory in selected[1:]:
            data=torch.load(directory/"datasets.pt",weights_only=True)
            for split in first:
                for key in first[split]:
                    torch.testing.assert_close(first[split][key],data[split][key],rtol=0,atol=0)
