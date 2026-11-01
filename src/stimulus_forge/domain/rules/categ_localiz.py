
import os
import glob
import pandas as pd
import traceback
from collections import OrderedDict
from stimulus_forge.domain.rules.task_info_base import TaskInfoBase

class CategLocalizInfo(TaskInfoBase):
    def __init__(self, custom_pics_path=None):
        super().__init__(custom_pics_path)
        self.task_folder = os.path.dirname(custom_pics_path)
        if os.path.exists(self.task_folder):
            self.read_img_info_table(self.task_folder)

    def read_img_info_table(self, task_folder, task_group='All'):
        xls_file = glob.glob(os.path.join(task_folder, '*.xlsx'))

        img_info_table = pd.read_excel(xls_file[0], sheet_name='AllCat', dtype=str)
        if task_group == 'RecDec':
            img_info_table = img_info_table[img_info_table.TaskGroup == 'RecDec']
        elif task_group == 'Other':
            img_info_table = img_info_table[img_info_table.TaskGroup == 'Other']

        img_names_list = [f"{x.lower()}-{y}.jpg" for x, y in zip(img_info_table.StimCat, img_info_table.StimNum)]
        img_info_table = pd.DataFrame({'name': img_names_list}).join(img_info_table)
        # add path column, find each image in the task folder
        img_paths = []
        for img_name in img_names_list:
            img_path = os.path.join(task_folder, img_name.split('-')[0], img_name)
            img_paths.append(img_path)

        
        img_info_table['used'] = False
        img_info_table['trial_count'] = 0
        img_info_table['path'] = img_paths
        img_info_table.reset_index(drop=True, inplace=True)
        self.img_info_table = img_info_table

    #region Rules for selecting stimuli for next subscreening

    def get_same_feats(self, stim_name):
        # Find the row for the selected stimulus
        stim_row = self.img_info_table[self.img_info_table['name'] == stim_name]
        if stim_row.empty:
            return []
        stim_cat = stim_row.iloc[0]['StimCat']
        # Get columns that start with stim_cat (feature columns)
        feat_cols = [col for col in self.img_info_table.columns if col.startswith(stim_cat)]
        # Build a mask for same features and not used
        mask = (self.img_info_table['used'] == False)
        for col in feat_cols:
            mask &= (self.img_info_table[col] == stim_row.iloc[0][col])
        # Exclude the original stimulus
        mask &= (self.img_info_table['name'] != stim_name)
        same_feats = self.img_info_table[mask]['name'].tolist()
        # Mark these as used
        self.img_info_table.loc[self.img_info_table['name'].isin(same_feats), 'used'] = True
        return same_feats
    
    def get_diff_feats(self, stim_name):
        # Find the row for the selected stimulus
        stim_row = self.img_info_table[self.img_info_table['name'] == stim_name]
        if stim_row.empty:
            return []
        stim_cat = stim_row.iloc[0]['StimCat']
        # Only consider rows with the same StimCat
        same_cat_table = self.img_info_table[self.img_info_table['StimCat'] == stim_cat]
        # Get columns that start with stim_cat (feature columns)
        feat_cols = [col for col in self.img_info_table.columns if col.startswith(stim_cat)]
        # Build a mask for different features and not used
        mask = (same_cat_table['used'] == False)
        for col in feat_cols:
            mask &= (same_cat_table[col] != stim_row.iloc[0][col])
        # Exclude the original stimulus
        mask &= (same_cat_table['name'] != stim_name)
        diff_feats = same_cat_table[mask]['name'].tolist()
        # Mark these as used in the main table
        self.img_info_table.loc[self.img_info_table['name'].isin(diff_feats), 'used'] = True
        return diff_feats
    
    def get_all_but_x_feats(self, stim_name, nonfixed_feats_count=1):
        # Find the row for the selected stimulus
        stim_row = self.img_info_table[self.img_info_table['name'] == stim_name]
        if stim_row.empty:
            return []
        stim_cat = stim_row.iloc[0]['StimCat']
        # Get columns that start with stim_cat (feature columns)
        feat_cols = [col for col in self.img_info_table.columns if col.startswith(stim_cat)]
        num_cols = len(feat_cols)
        if num_cols <= nonfixed_feats_count:
            return []
        from itertools import combinations
        cols_to_fix = list(combinations(range(num_cols), num_cols - nonfixed_feats_count))
        mask_total = (self.img_info_table['used'] == False)
        mask_same_feats = mask_total.copy()
        for col in feat_cols:
            mask_same_feats &= (self.img_info_table[col] == stim_row.iloc[0][col])
        mask_without_same_feats = ~mask_same_feats
        mask_all_but_x = [False] * len(self.img_info_table)
        for fix_combo in cols_to_fix:
            mask_fixed = mask_total.copy()
            for i in fix_combo:
                mask_fixed &= (self.img_info_table[feat_cols[i]] == stim_row.iloc[0][feat_cols[i]])
            non_fixed_cols = [i for i in range(num_cols) if i not in fix_combo]
            mask_nonfixed = mask_total.copy()
            for i in non_fixed_cols:
                mask_nonfixed &= (self.img_info_table[feat_cols[i]] != stim_row.iloc[0][feat_cols[i]])
            mask_combo = mask_fixed & mask_nonfixed
            mask_all_but_x = [a or b for a, b in zip(mask_all_but_x, mask_combo)]
        # Remove same_feats
        mask_final = mask_without_same_feats & mask_all_but_x
        # Exclude the original stimulus
        mask_final &= (self.img_info_table['name'] != stim_name)
        all_but_x_feats = self.img_info_table[mask_final]['name'].tolist()
        # Mark these as used
        self.img_info_table.loc[self.img_info_table['name'].isin(all_but_x_feats), 'used'] = True
        return all_but_x_feats

    #endregion Rules for selecting stimuli for next subscreening
    

    # get_nxt_scr_stim_list is inherited from TaskInfoBase
    
    def execute_rules(self, scr_id, scr_selected_stims, verbose=True):
        scr_stim_list = []
        same_units = {}
        diff_units = {}

        for stim_name in scr_selected_stims:
            same_units[stim_name] = self.get_same_feats(stim_name)
            diff_units[stim_name] = self.get_diff_feats(stim_name)
        scr_stim_list.extend([item for sublist in same_units.values() for item in sublist])
        scr_stim_list.extend([item for sublist in diff_units.values() for item in sublist])
        scr_stim_list = list(dict.fromkeys(scr_stim_list))
        self.task_info["subscr_selections"]["subscreening_"+str(scr_id)] = [
            {
                "rule_name":"selected_stims",
                "pics_added": scr_selected_stims
            },
            {
                "rule_name":"same_feats",
                "pics_added": same_units
            },
            {
                "rule_name":"diff_feats",
                "pics_added": diff_units
            }]
        if verbose:
            self.print_subscreening_info(scr_id, self.task_info["subscr_selections"]["subscreening_"+str(scr_id)])
        scr_stim_list = self.get_ordered_stim_list()
        return scr_stim_list
    
    # get_ordered_stim_list is inherited from TaskInfoBase
    
    # print_subscreening_info is inherited from TaskInfoBase
