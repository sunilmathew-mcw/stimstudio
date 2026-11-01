import os
import random
import pandas as pd
import traceback
from collections import OrderedDict

class TaskInfoBase:
    
    def __init__(self, block_params, exp_params,custom_pics_path=None):
        self.block_params = block_params
        self.exp_params = exp_params
        self.custom_pics_path = custom_pics_path
        self.img_info_table = None
        self.max_trial_stims = {}
        self.selected_stims = {}
        self.task_info = {"subscr_selections": {}}
        self.subscr_sel_info_txt = {}
        self.task_name = "base_task"
        self.task_stims_folder = os.path.dirname(custom_pics_path) if custom_pics_path else None

    def save_task_info(self, save_folder):
        import json
        try:
            # Save img_info_table to CSV
            if self.img_info_table is not None:
                img_info_path = os.path.join(save_folder, "img_info_table.csv")
                self.img_info_table.to_csv(img_info_path, index=False)
            # Save task_info to JSON
            task_info_path = os.path.join(save_folder, "task_info.json")
            with open(task_info_path, "w") as f:
                json.dump(self.task_info, f, indent=4)
            # Save selected_stims to JSON
            selected_stims_path = os.path.join(save_folder, "selected_stims.json")
            with open(selected_stims_path, "w") as f:
                json.dump(self.selected_stims, f, indent=4)
            # Save init params
            params = {
                "block_params": self.block_params,
                "exp_params": self.exp_params,
                "custom_pics_path": self.custom_pics_path
            }
            params_path = os.path.join(save_folder, "init_params.json")
            with open(params_path, "w") as f:
                json.dump(params, f, indent=4)
        except Exception:
            print(traceback.format_exc())

    @classmethod
    def load_task_info(cls, load_folder):
        import json
        obj = None
        try:
            # Load init params
            with open(os.path.join(load_folder, "init_params.json"), "r") as f:
                params = json.load(f)
            obj = cls(**params)
            # Load img_info_table
            img_info_path = os.path.join(load_folder, "img_info_table.csv")
            obj.img_info_table = pd.read_csv(img_info_path)

            try:
                # Load task_info
                task_info_path = os.path.join(load_folder, "task_info.json")
                with open(task_info_path, "r") as f:
                    obj.task_info = json.load(f)
            except:
                obj.task_info = {"subscr_selections": {}}
            # Load selected_stims
            selected_stims_path = os.path.join(load_folder, "selected_stims.json")
            with open(selected_stims_path, "r") as f:
                obj.selected_stims = json.load(f)
            return obj
        except Exception:
            print(traceback.format_exc())
            return obj
    
    def get_nxt_scr_stim_list(self, scr_id, scr_selected_stims, discarded_stimuli, 
                              scr_responsive_stims, prev_subscreening=None, 
                              update_table=False, verbose=True):
        extra_stims = []
        removed_stims = []
        if self.img_info_table is None:
            return []
        scr_fetched_list = self.execute_rules(scr_id, scr_selected_stims, update_table=update_table, verbose=verbose) # will have all selected + rule based added ordered by subscreening
        # Remove any that reached max_trials
        max_trial_stims = self.max_trial_stims.get(prev_subscreening.scr_id, [])
        scr_fetched_list = [stim for stim in scr_fetched_list if stim not in max_trial_stims]
        scr_responsive_stims = [stim for stim in scr_responsive_stims if stim not in max_trial_stims]
        p2remove = self.block_params.get(f"block{prev_subscreening.scr_id}", {}).get("p2remove", 0)
        if p2remove > 0:
            p2remove = p2remove - (len(discarded_stimuli) + len(max_trial_stims)) if len(discarded_stimuli) < p2remove else 0
            if p2remove > 0:
                # Ensure p2remove does not exceed the list length
                p2remove = min(p2remove, len(scr_responsive_stims))
                valid_responsive_stims = scr_responsive_stims[:len(scr_responsive_stims)-p2remove]  # remove p2remove stims from the end
                removed_stims.extend(scr_responsive_stims[len(scr_responsive_stims)-p2remove:])  # remove p2remove stims from the end
        if len(scr_fetched_list) > self.block_params[f"block{scr_id}"]["n_pics"]:
            scr_fetched_list = scr_fetched_list[:self.block_params[f"block{scr_id}"]["n_pics"]]
            extra_stims = scr_fetched_list[self.block_params[f"block{scr_id}"]["n_pics"]:]
            extra_stims.extend(valid_responsive_stims)
        elif len(scr_fetched_list) + len(valid_responsive_stims) > self.block_params[f"block{scr_id}"]["n_pics"]:
            valid_responsive_stims = valid_responsive_stims[:self.block_params[f"block{scr_id}"]["n_pics"] - len(scr_fetched_list)]
            scr_fetched_list.extend(valid_responsive_stims)
            extra_stims = valid_responsive_stims[self.block_params[f"block{scr_id}"]["n_pics"] - len(scr_fetched_list):]
        elif len(scr_fetched_list) + len(valid_responsive_stims) < self.block_params[f"block{scr_id}"]["n_pics"]:
            scr_fetched_list.extend(valid_responsive_stims)
            # Add random stims to fill up
            random_needed = self.block_params[f"block{scr_id}"]["n_pics"] - len(scr_fetched_list)
            random_stims = self.get_random_stims(random_needed=random_needed,
                                                scr_fetched_list=scr_fetched_list,
                                                scr_id=scr_id)
            if random_stims:
                scr_fetched_list.extend(random_stims)
                if update_table:
                    self.img_info_table.loc[self.img_info_table["name"].isin(random_stims), "status"] = "active"
                    self.img_info_table.loc[self.img_info_table["name"].isin(random_stims), "rule_name"] = "random"
        
        scr_fetched_list = list(dict.fromkeys(scr_fetched_list))
        if update_table:
            self.img_info_table.loc[self.img_info_table["name"].isin(discarded_stimuli), "status"] = "discarded"
            self.img_info_table.loc[self.img_info_table["name"].isin(removed_stims), "status"] = "removed"

        if len(scr_fetched_list) != self.block_params[f"block{scr_id}"]["n_pics"]:
            print(f"Warning: Fetched {len(scr_fetched_list)} pics for subscreening {scr_id}, expected {self.block_params[f'block{scr_id}']['n_pics']}")

        return scr_fetched_list, extra_stims, removed_stims
    
    def get_random_stims(self, random_needed, scr_fetched_list, scr_id):
        random_stims = []
        if self.img_info_table is None or random_needed <= 0:
            return
        # Get stims with status "not_shown"
        not_shown_stims = self.img_info_table[self.img_info_table["status"] == "not_shown"]["name"].values.tolist()
        # Exclude those already in scr_fetched_list
        not_shown_stims = [stim for stim in not_shown_stims if stim not in scr_fetched_list]
        if not not_shown_stims:
            return
        if len(not_shown_stims) <= random_needed:
            random_stims = not_shown_stims
            print(f"Only {len(not_shown_stims)} not_shown stims available, needed {random_needed} for subscreening {scr_id}. Adding all available.")
        else:
            # Randomly select from not_shown_stims
            random_stims = random.sample(not_shown_stims, random_needed)

        return random_stims

    def update_stim_trial_count(self, prev_subscreening, seq_idx=None):
        """ Update the trial_count and status of stimuli based on previous subscreening results 
            and updates status, rule_name
            Returns list of stimuli that reached max_trials
        """
        max_trial_stims = []
        shown_stims = []
        if self.img_info_table is None or prev_subscreening is None:
            return
        if hasattr(prev_subscreening, 'b_aborted') and prev_subscreening.b_aborted:
            last_valid_seq = prev_subscreening.last_valid_seq
        else:
            last_valid_seq = getattr(prev_subscreening, 'seq_count', 0) - 1

        

        if not prev_subscreening.b_last_subscr and not prev_subscreening.b_aborted and seq_idx is None:    
            # stims shown in this subscreening
            shown_stims = prev_subscreening.stim_list['name'].values.tolist()        
            shown_mask = self.img_info_table["name"].isin(shown_stims)
            # Increment trial_count for shown_stims n_rep times
            self.img_info_table.loc[shown_mask, "trial_count"] += prev_subscreening.n_rep
            
        else:
            sequences = getattr(prev_subscreening, 'sequences', [])
            if seq_idx is not None:
                seqs_to_process = [sequences[seq_idx]]
            else:
                seqs_to_process = sequences[:last_valid_seq+1]
            for seq in seqs_to_process:
                for stim in seq:
                    if stim not in shown_stims:
                        shown_stims.append(stim)
                    self.img_info_table.loc[self.img_info_table["name"] == stim, "trial_count"] += 1

            shown_mask = self.img_info_table["name"].isin(shown_stims)
            
        # set status to active.
        self.img_info_table.loc[shown_mask, "status"] = "active"
        # append to scr_shown scr_id
        self.img_info_table.loc[shown_mask, "scr_shown"] = self.img_info_table.loc[shown_mask, "scr_shown"].apply(
            lambda x: x + [prev_subscreening.scr_id] if isinstance(x, list) and prev_subscreening.scr_id not in x
            else x if isinstance(x, list)
            else [x, prev_subscreening.scr_id] if pd.notnull(x) and x != prev_subscreening.scr_id
            else [prev_subscreening.scr_id]
        )
        if prev_subscreening.scr_id == 1:
            self.img_info_table.loc[shown_mask, "rule_name"] = "custom"
        # Check if any have reached max_trials
        self.img_info_table.loc[self.img_info_table["trial_count"] >= self.exp_params["max_trials"], "status"] = "max_trials"
        max_trial_stims = self.img_info_table[self.img_info_table["status"] == "max_trials"]["name"].values.tolist()
        self.max_trial_stims[prev_subscreening.scr_id] = max_trial_stims

    def update_discarded_stims(self, discarded_stims):
        if self.img_info_table is None or not discarded_stims:
            return
        for stim in discarded_stims:
            if stim in self.img_info_table["name"].values:
                self.img_info_table.loc[self.img_info_table["name"] == stim, "status"] = "discarded"

    def execute_rules(self, scr_id, scr_selected_stims, verbose=True):
        # Subclasses should implement their own rule logic
        raise NotImplementedError()

    def get_ordered_stim_list(self):
        def get_pics_added(rule):
            if isinstance(rule["pics_added"], list):
                return rule["pics_added"]
            elif isinstance(rule["pics_added"], dict):
                pics = []
                for stim, pic_list in rule["pics_added"].items():
                    pics.extend(pic_list)
                return pics
            else:
                return []
        ordered_stim_list = []
        all_stimgs = OrderedDict()
        for subscr_id, subscr_info in self.task_info["subscr_selections"].items():
            for rule in subscr_info:
                if rule["rule_name"] not in all_stimgs:
                    all_stimgs[rule["rule_name"]] = get_pics_added(rule)
                else:
                    all_stimgs[rule["rule_name"]].extend(get_pics_added(rule))
        for pics in all_stimgs.values():
            ordered_stim_list.extend(pics)
        ordered_stim_list = list(dict.fromkeys(ordered_stim_list))
        return ordered_stim_list

    def print_subscreening_info(self, scr_id, subscr_info):
        subscr_info_txt = f"Subscreening {scr_id-1}:\n"
        subscr_info_txt += f"Selected stimuli({len(subscr_info[0]['pics_added'])}): {', '.join(subscr_info[0]['pics_added'])} \n"
        added_count = len(subscr_info[0]['pics_added'])
        for rule in subscr_info[1:]:
            if isinstance(rule['pics_added'], dict):
                added_count = sum(len(pics) for pics in rule['pics_added'].values())
                subscr_info_txt += f"Rule: {rule['rule_name']} - added {added_count} pics \n"
                for stim, pics in rule['pics_added'].items():
                    subscr_info_txt += f"{stim} -> ({', '.join(pics)}) \n"
            elif isinstance(rule['pics_added'], list):
                added_count = len(rule['pics_added'])
                subscr_info_txt += f"Rule: {rule['rule_name']} - added {added_count} pics \n"
                subscr_info_txt += f"({', '.join(rule['pics_added'])}) \n"
        print(subscr_info_txt)
        self.subscr_sel_info_txt[f"subscreening_{scr_id-1}"] = subscr_info_txt

    def load_img_info(self, folder_path):
        """ Load the img_info_table from file
        """
        img_info_table_path = os.path.join(folder_path, 'img_info_table.csv')
        if os.path.exists(img_info_table_path):
            self.img_info_table = pd.read_csv(img_info_table_path)