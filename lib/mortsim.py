import pandas as pd
import numpy as np

class ASMortSim:

    def __init__(self, params):
        self._params = params

    def run(self):
        
        # get the simulation parameters
        curr_size = self._params.population_size
        curr_iss_age = self._params.issue_age
        level_term_length = self._params.level_period_length
        post_level_length = self._params.post_level_period_length
        pct_as = self._params.true_as_w
        shock_lapse_rate = self._params.shock_lapse
        mort_table = self._params.mort_table
        num_sim_years = post_level_length + 1

        # create a vector of ages (starting @ duration before PLT)
        curr_ages =  curr_iss_age * np.ones((int(curr_size), 1))\
            + (level_term_length - 1)
        
        # generate the class (mixture) probabilities
        class_probs = np.array([
            1 - pct_as,
            pct_as])
        
        # generate the splits between non-as / as
        class_assignments = np.random.multinomial(1, class_probs, size=int(curr_size))

        # create a mortality assumption for the last year of the level
        # term and the first year of the post-level    
        ult_ages = [curr_iss_age + level_term_length + i for i in range(0,6)]
        qx_mat = np.array([
            mort_table.loc[ult_ages, "q_x"],
            mort_table.loc[ult_ages, "q_as"]
        ])

        # get the **actual** mortality for each record
        qx_actual = class_assignments @ qx_mat

        # get the *shock* lapse rates for each record
        shock_lapse_probs = np.array([
            shock_lapse_rate,
            0.0
        ])

        # prepare a matrix to be used in the next step for lapse rates
        w_mat = np.concatenate([
            np.tile(shock_lapse_probs.reshape(-1, 1), num_sim_years)
        ], axis=1)
        
        # match up the lapse rates with the correct class
        w_actual = class_assignments @ w_mat

        # random values for the lapse decrements
        lapse_sim_vals = np.random.uniform(size=w_actual.shape)

        # simulate lapses
        # determine the year that the lapse decrement occures 
        lapse_sim = np.argmin(np.concatenate([
            (np.floor(lapse_sim_vals - w_actual)),
            # add a sentinel value to the end so that if the decrement doesn't occur
            # the argmin returns a value past the end of the simulation
            -1 * np.ones(shape=(w_actual.shape[0], 1))
        ], axis=1), axis=1)

        # create a matrix w/ indicator values for the year that the lapse occured
        lapse_sim_ident = np.identity(num_sim_years + 1)[lapse_sim].astype('int')
        
        # random values for the death decrements
        death_sim_vals = np.random.uniform(size=qx_actual.shape)

        # simulate deaths
        # determine the year that the death decrement occurs
        death_sim = np.argmin(np.concatenate([
            (np.floor(death_sim_vals - qx_actual)),
            # add a sentinel value to the end so that if the decrement doesn't occur
            # the argmin returns a value past the end of the simulation
            -1 * np.ones(shape=(death_sim_vals.shape[0], 1))
        ], axis=1), axis=1)

        # create a matrix with indicator values for the year that death occured
        death_sim_ident = np.identity(num_sim_years + 1)[death_sim].astype('int')

        # bit masks for use in decrement ordering, i.e.,
        # death or lapse can only happen once per life and
        # deaths happen prior to lapses
        death_bit_mask = 1 - np.cumsum(death_sim_ident, axis=1)
        lapse_bit_mask = 1 - np.cumsum(lapse_sim_ident, axis=1)

        # indicator if policy inforce at the end of the year
        inforce_eoy = np.bitwise_and(death_bit_mask, lapse_bit_mask)
        
        # indicator if policy inforce at the end of the year
        inforce_boy = np.concatenate([
            np.ones((inforce_eoy.shape[0], 1)),
            inforce_eoy[:,0:-1]
        ], axis=1).astype('int')

        # final death counts: must be inforce at BOY to be a death
        death_sim_final = np.bitwise_and(
            inforce_boy,
            death_sim_ident
        )

        # final lapse counts: must be inforce at BOY and not die during 
        # year to be a lapse
        lapse_sim_final = np.bitwise_and(
            inforce_boy,
            np.bitwise_and(1 - death_sim_final, lapse_sim_ident)
        )

        # assume that lapses (not deaths) are the final decrement
        lapse_sim_final[:,-2] = lapse_sim_final[:,-2] + death_sim_final[:,-1]  

        # remove the last column of each array (since past the end of sim)
        lapse_sim_final = lapse_sim_final[: , 0:-1]
        death_sim_final = death_sim_final[: , 0:-1]

        # compute return values
        total_eoy_exits =  np.cumsum(
            np.sum(lapse_sim_final, axis=0) + np.sum(death_sim_final, axis=0))
        
        boy_inforce_final = np.roll(int(curr_size) - total_eoy_exits, 1)
        boy_inforce_final[0] = int(curr_size)

        # compute the number of inforce that is anti-selective
        as_class_mask = np.tile(class_assignments[:,1].reshape(-1, 1), num_sim_years).astype('int')
        as_inforce_boy = np.bitwise_and(
            inforce_boy[:,0:-1],
            as_class_mask
        )

        # compute the number of deaths that are anti-selective
        as_deaths = np.bitwise_and(
            death_sim_final,
            as_class_mask
        )

        return {            
            "issue_age": np.repeat(curr_iss_age, num_sim_years),
            "attained_age": ult_ages,
            "policy_year": np.array(ult_ages) - curr_iss_age,
            "boy_inforce": boy_inforce_final,
            "total_deaths": np.sum(death_sim_final, axis=0),
            "total_lapses": np.sum(lapse_sim_final, axis=0),
            "boy_inforce_as": np.sum(as_inforce_boy, axis=0),
            "total_deaths_as": np.sum(as_deaths, axis=0),
            "qx_norm": qx_mat[0,:],
            "qx_as": qx_mat[1,:]
        }

class ASMortSimParams():

    def __init__(self, non_as_mort_table, **kwargs):

        # use some defaults
        self.true_as_qx = 0.30 #@param {type:"number"}
        self.true_as_w = 0.035 #@param {type:"number"}
        self.issue_age = 35 #@param {type:"integer"}
        self.population_size = 5000 #@param {type:"integer"}
        self.shock_lapse = 0.6 #@param {type:"number"}
        self.level_period_length = 20 #@param {type:"integer"}
        self.post_level_period_length = 5 #@param {type:"integer"}

        # setup the mort table
        self.mort_table = non_as_mort_table.copy()
        self.mort_table["q_as"] = self.true_as_qx

        # update defaults
        self.__dict__.update(kwargs)

    @staticmethod
    def load_ult_mort_table(file_path):
        expected_schema = ["attained_age","q_x"]
        mort_table = pd.read_csv(file_path)
        if not list(mort_table.columns) == expected_schema:
            raise Exception(f"Unexpected schema, expected {expected_schema}")

        mort_table = mort_table.set_index("attained_age")
        return mort_table