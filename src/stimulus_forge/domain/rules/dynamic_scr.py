# class to dynamically get stimuli that is related to responsive stimuli for subsequent subscreening

import random
import pandas as pd
import traceback
import os
from collections import OrderedDict
from .task_info_base import TaskInfoBase

class DynamicScrInfo(TaskInfoBase):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(DynamicScrInfo, cls).__new__(cls)
        return cls.instance
    
    def __init__(self, block_params, exp_params, custom_pics_path=None):
        super().__init__(block_params=block_params, exp_params=exp_params, custom_pics_path=custom_pics_path)
        self.image_paths = self.get_task_imgs(self.task_stims_folder)
        self.img_info_table = self.create_img_info_table()
        self.task_name = "dynamic_scr"    

    def get_task_imgs(self, folder_path=None):
        try:
            if folder_path is None:
                return
            if not os.path.exists(folder_path):
                print(f"[Error] {folder_path} does not exist.")
                return
            # Walk through the folder and get all images
            images = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        images.append(os.path.join(root, file))

            return images
        except:
            print(traceback.format_exc())

    def create_img_info_table(self):
        try:
            if self.image_paths is None:
                return pd.DataFrame(columns=["name", "scr_shown", "trial_count", 
                                             "status", # active, discarded, not_shown, max_trials
                                             "rule_name", # custom, selected, same_units, related_units
                                             "concept", "concept_num", "concepts", "path"])     
            # Prepare data for DataFrame
            rows = []
            for img_path in self.image_paths:
                image_name = os.path.basename(img_path)
                if "_" in image_name:
                    info = image_name.split(".")[0]
                    concept_num = info.split("_")[1]
                    concept = info.split("_")[0]
                    if "~" in image_name:
                        concepts = concept.split("~")
                    else:
                        concepts = [concept]
                    rows.append({
                        "name": image_name,
                        "scr_shown": [], # list of subscreenings where this image was shown
                        "trial_count": 0,
                        "status": "not_shown", # can be "not_shown", "selected", "active", "discarded", "max_trials"
                        "rule_name": "not_shown", # can be "custom", "same_units", "related_units"
                        "concept": concept,
                        "concept_num": concept_num,
                        "concepts": concepts,
                        "path": img_path
                    })

            if rows:
                df = pd.DataFrame(rows)
                return df
            else:
                return pd.DataFrame(columns=["name", "scr_shown", "concept", "concept_num", "concepts", "trial_count"])
        except Exception:
            print(traceback.format_exc())

    

    #region Rules for selecting stimuli for next subscreening

    def get_same_units(self, unit):
        same_units = []
        # Get rows that have the same concept
        if self.img_info_table is None:
            return same_units
        
        concept = unit.split("_")[0] if "_" in unit else unit

        # get not shown images with the same concept
        same_units_rows = self.img_info_table[(self.img_info_table["concept"] == concept) & 
                                              (self.img_info_table["status"] == "not_shown") &
                                              (self.img_info_table["name"] != unit)]
        if not same_units_rows.empty:
            same_units = same_units_rows["name"].values.tolist()
            # if more than 2 same units, randomly select 2
            if len(same_units) > 2:
                same_units = random.sample(same_units, 2)
        return same_units

    def get_related_units(self, same_units, img_name):
        same_cat_imgs = []
        # Get rows that are related by seeing if any of the concepts match
        if self.img_info_table is None:
            return same_cat_imgs

        concept = img_name.split("_")[0] if "_" in img_name else img_name

        img_concepts = self.img_info_table[self.img_info_table["name"] == img_name]["concepts"].values
        if not img_concepts:
            return same_cat_imgs

        # Flatten img_concepts in case it's an array of arrays
        img_concepts_set = set()
        for c in img_concepts:
            img_concepts_set.update(c)

        # Check for any overlap between concepts columns and img_concepts_set and status is "not_shown"
        not_shown_mask = self.img_info_table["status"] == "not_shown"
        img_info_table = self.img_info_table[not_shown_mask]
        # remove same_units from img_info_table
        img_info_table = img_info_table[~img_info_table["name"].isin(same_units)]
        # get rows where any of the concepts match
        related_units = img_info_table[img_info_table["concepts"].apply(lambda x: bool(set(x) & img_concepts_set))]
        related_units = related_units[related_units["name"] != img_name]["name"].values.tolist()
        # if more than 2 related units, randomly select 2
        if len(related_units) > 2:
            related_units = random.sample(related_units, 2)
        return related_units
    
    #endregion Rules for selecting stimuli for next subscreening

    # get_nxt_scr_stim_list is inherited from TaskInfoBase

    # update_stim_trial_count is inherited from TaskInfoBase

    def execute_rules(self, scr_id, scr_selected_stims, update_table=False, verbose=True):
        scr_stim_list = []
        same_units = {}
        related_units = {}
        for stim_name in scr_selected_stims:
            same_units[stim_name] = self.get_same_units(stim_name)
        same_units_flat = [item for sublist in same_units.values() for item in sublist]
        
        for stim_name in scr_selected_stims:
            related_units[stim_name] = self.get_related_units(same_units=same_units_flat, img_name=stim_name)
        
        related_units_flat = [item for sublist in related_units.values() for item in sublist]
        if update_table:
            self.selected_stims[f"subscreening_{scr_id-1}"] = scr_selected_stims
            self.img_info_table.loc[self.img_info_table["name"].isin(scr_selected_stims), "status"] = "selected"
            self.img_info_table.loc[self.img_info_table["name"].isin(same_units_flat), "status"] = "active"
            self.img_info_table.loc[self.img_info_table["name"].isin(same_units_flat), "rule_name"] = "same_units"
            self.img_info_table.loc[self.img_info_table["name"].isin(related_units_flat), "status"] = "active"
            self.img_info_table.loc[self.img_info_table["name"].isin(related_units_flat), "rule_name"] = "related_units"
        self.task_info["subscr_selections"]["subscreening_"+str(scr_id-1)] = [
            {
                "rule_name":"selected_stims",
                "pics_added": scr_selected_stims
            },
            {
                "rule_name":"same_units",
                "pics_added": same_units
            },
            {
                "rule_name":"related_units",
                "pics_added": related_units
            }]
        if verbose:
            self.print_subscreening_info(scr_id, self.task_info["subscr_selections"]["subscreening_"+str(scr_id-1)])
        scr_stim_list = self.get_ordered_stim_list()
        return scr_stim_list
    
    
    # get_ordered_stim_list is inherited from TaskInfoBase
    
    # print_subscreening_info is inherited from TaskInfoBase
