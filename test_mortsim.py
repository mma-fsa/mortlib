import unittest
from lib.mortsim import ASMortSim, ASMortSimParams

class Testing(unittest.TestCase):
   
    def test_canary(self):
        self.assertTrue(True)

    def test_param_constructor(self):

        mort_tbl = ASMortSimParams.load_ult_mort_table("./2001_VBT_Residual_Standard_Ultimate_Male_Nonsmoker_ANB.csv")
        self.assertIsNotNone(mort_tbl)

        default = ASMortSimParams(mort_tbl)
        self.assertIsNotNone(default.true_as_qx)
        self.assertIsNotNone(default.true_as_w)
        self.assertIsNotNone(default.issue_age)
        self.assertIsNotNone(default.population_size)
        self.assertIsNotNone(default.shock_lapse)
        self.assertIsNotNone(default.level_period_length)        
        self.assertIsNotNone(default.post_level_period_length)
        
        override = ASMortSimParams(mort_tbl, true_as_qx=0.1)
        self.assertAlmostEquals(override.true_as_qx, 0.1)

    def test_simulation(self):
        mort_tbl = ASMortSimParams.load_ult_mort_table("./2001_VBT_Residual_Standard_Ultimate_Male_Nonsmoker_ANB.csv")
        default = ASMortSimParams(mort_tbl)
        sim = ASMortSim(default)

        sim_results = sim.run()
    
if __name__ == '__main__':
    unittest.main()