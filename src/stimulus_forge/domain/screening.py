# Author: Sunil Mathew
# Creation Date: 9/5/2025
import numpy as np
import random
import traceback
from time import gmtime, strftime

class SubscreeningConfig:
    def __init__(self, id, task_name, stim_list, 
                 n_rep=1, isi=0.5, max_trials=12, 
                 sort_n_rep=1,
                 b_manual_select=False, b_last=False):
        self.scr_id = id
        self.id = f'subscreening_{id}'
        self.task_name = task_name
        self.stim_list = stim_list # pandas dataframe with id, name, concepts, related_concepts, trial_count
        self.isi = isi # ISI in seconds
        self.min_seq_length = np.ceil(30 / isi).astype(int)  # Minimum sequence length of 30 seconds
        self.seq_count = None
        self.b_aborted = False
        self.n_rep = n_rep
        self.sort_n_rep = sort_n_rep
        self.sort_seq_id = 0
        self.max_trials = max_trials
        self.b_manual_select = b_manual_select
        self.b_last_subscr = b_last
        self.aborted_seq = -1
        self.aborted_trial = -1
        self.last_valid_seq = -1

        self.seq_length = None
        self.sequences = []
        self.line_changes = []  
        self.stim_on_times = []  # List of lists, each sublist corresponds to a sequence
        self.create_sequences()
        self.generate_line_changes()

    def generate_line_changes(self, nmean_ch=3):
        """
        Generate line_changes for each sequence (rep), matching the structure of self.sequences.
        Each entry in line_changes corresponds to a sequence and contains only the pic changes for that sequence.
        Each row: [ISI rank, time, top color, bottom color, pic index].
        """
        n_seq = self.seq_count or 0
        seq_length = self.seq_length or 0
        ISI = self.isi
        color_options = ["red", "yellow"]
        self.line_changes = []
        self.line_changes_seq_start = []

        for irep in range(n_seq):
            begin_color_top = random.choice(color_options)
            begin_color_bottom = color_options[1] if begin_color_top == color_options[0] else color_options[0]
            self.line_changes_seq_start.append({'top_color': begin_color_top, 'bottom_color': begin_color_bottom})
            # Pic changes: random number of changes per sequence
            nchanges = nmean_ch + random.randint(-1, 1)
            # Pick unique, sorted change points (avoid first/last pic)
            possible_pics = list(range(2, seq_length))
            change_pics = sorted(random.sample(possible_pics, min(nchanges, len(possible_pics))))

            # Build color sequence for top/bottom, no consecutive repeats
            color_seq = []
            last_color = begin_color_top
            for _ in range(len(change_pics)):
                next_color = color_options[0] if last_color != color_options[0] else color_options[1]
                color_seq.append(next_color)
                last_color = next_color

            pic_changes = {}
            for idx, pic_idx in enumerate(change_pics):
                isi_rank = 1
                delay = round(random.uniform(ISI/4, 3*ISI/4), 3)
                top_color = color_seq[idx % len(color_seq)]
                bottom_color = color_options[1] if top_color == color_options[0] else color_options[0]
                # pic_changes.append([isi_rank, time_change, top_color, bottom_color, pic_idx])
                pic_changes[pic_idx] = {'isi_rank': isi_rank, 'delay': delay, 'top_color': top_color, 'bottom_color': bottom_color}
            

            self.line_changes.append(pic_changes)
        
        # Debug: print line_changes
        # for seq_idx, changes in enumerate(self.line_changes):
        #     print(f"Subscr {self.id}: Sequence {seq_idx}: {changes}\n")

    def create_sequences(self):
        """
        Create sequences for subscreening
        """
        try:
            if not self.b_last_subscr:
                self.sequences = self.create_sequences_normal()
            else:
                self.sequences = self.create_sequences_last()
                if len(self.sequences) == 0:
                    print(f"Warning: No stimuli available for last subscreening {self.id}. Check stim_list and max_trials.")
            self.stim_on_times = [[] for _ in range(len(self.sequences))]  # Initialize stim_on_times for each sequence
        except:
            print(traceback.format_exc())

    def create_sequences_normal(self):
        """
        Create sequences for subscreening when it is not the last screening
        Goals is to repeat the stimuli n_rep times, not show the same stimulus in the same sequence,
        and have sequences of at least min_seq_length
        """
        screening_list = []
        stim_names = self.stim_list['name'].tolist() # list of stimulus names that have not been shown max_trials times        
            
        stim_count = len(stim_names)
        total_trials = stim_count * self.n_rep
        if stim_count < self.min_seq_length:
            self.seq_length = stim_count * 2  # Each sequence shows each stimulus twice
            self.n_rep = self.n_rep + np.mod(self.n_rep, 2)  # Make n_rep even first
            self.seq_count = self.n_rep // 2  # Each sequence provides 2 reps, so need n_rep/2 sequences
            n_seq_per_rep = 0.5  # Each sequence provides 2 reps
        else:
            n_seq_per_rep = np.floor(stim_count / self.min_seq_length).astype(int)
            self.seq_length = np.ceil(stim_count / n_seq_per_rep).astype(int)
            self.seq_count = self.n_rep * n_seq_per_rep

        self.sort_seq_id = self.sort_n_rep * n_seq_per_rep if self.sort_n_rep > 0 else 0

        for rep in range(self.n_rep):
            random.shuffle(stim_names)
            screening_list.extend(stim_names)

        # break into sequences
        screening_list = screening_list[:self.seq_length * self.seq_count]
        sequences = [screening_list[i * self.seq_length:(i + 1) * self.seq_length] for i in range(self.seq_count)]
        return sequences
    
    def create_sequences_last(self):
        """
        Create sequences for subscreening when it is the last screening
        The idea is to balance the number of trials across all stimuli
        that made to the last subscreening
        """
        screening_stim_list = []

        stim_trial_count_arr = np.unique(self.stim_list['trial_count'])
        stim_trial_count_arr = np.sort(stim_trial_count_arr)
        pics_groups = {}
        for idx in range(len(stim_trial_count_arr)):
            stim_trial_count = stim_trial_count_arr[idx]
            req_stim_trial_count = self.max_trials - stim_trial_count
            pics = self.stim_list[self.stim_list['trial_count'] == stim_trial_count]
            pics_groups[len(pics_groups) + 1] = {'pics': pics, 'req_stim_trial_count': req_stim_trial_count}

        if len(pics_groups) == 0:
            return []

        while pics_groups[1]['req_stim_trial_count'] > 0:
            pics_list = []
            for pics_group_idx in range(1, len(pics_groups) + 1):
                pics_group = pics_groups[pics_group_idx]
                if pics_group['req_stim_trial_count'] > 0:
                    pics = pics_group['pics']
                    pics_names = pics['name'].tolist()
                    random.shuffle(pics_names)
                    pics_list.extend(pics_names)
                    pics_groups[pics_group_idx]['req_stim_trial_count'] -= 1
            random.shuffle(pics_list)
            screening_stim_list.extend(pics_list)

        n_seq_rep = np.floor(len(screening_stim_list) / self.min_seq_length).astype(int)
        if n_seq_rep < 1:
            n_seq_rep = 1
        self.seq_length = np.ceil(len(screening_stim_list) / n_seq_rep).astype(int)

        self.seq_count = n_seq_rep
        screening_list = screening_stim_list[:self.seq_length * self.seq_count]
        sequences = [screening_list[i * self.seq_length:(i + 1) * self.seq_length] for i in range(self.seq_count)]
        return sequences
    
    def get_config_str(self):        
        config_str = self.get_config_summary() + "\n"
        stim_names = self.stim_list['name'].tolist()
        
        # config_str += f"Stimuli ({len(stim_names)}): " + ", ".join(stim_names) + "\n\n"
        config_str += f"Stimuli ({len(stim_names)}): " + ", ".join([f"{i+1}. {name}" for i, name in enumerate(stim_names)]) + "\n\n"
        
        for seq_idx in range(len(self.sequences)):
            # line changes for this sequence
            config_str += f"Line changes for Sequence {seq_idx + 1}"
            config_str += f"(begin_top:{self.line_changes_seq_start[seq_idx]['top_color']}, "
            config_str += f"begin_bottom:{self.line_changes_seq_start[seq_idx]['bottom_color']}): " + str(self.line_changes[seq_idx]) + "\n\n"
            # config_str += f"Sequence {seq_idx + 1}: {', '.join(self.sequences[seq_idx])} \n\n"
            config_str += f"Sequence {seq_idx + 1}: " + ", ".join([f"{i+1}. {name}" for i, name in enumerate(self.sequences[seq_idx])]) + " \n\n"

        return config_str
    
    def get_config_summary(self):
        subscreening_time = strftime("%H:%M:%S", gmtime(self.isi * self.seq_length * self.seq_count))
        config_str = (
            f"Subscreening {self.scr_id}: stim_count={self.stim_list.shape[0]}, "
            f"n_rep={self.n_rep}, isi={self.isi}, "
            f"seq_length={self.seq_length}, seq_count={self.seq_count}, "
            f"total_trials={self.seq_length * self.seq_count}, "
            f"subscreening time=~{subscreening_time} \n"
        )

        return config_str
    
    def _convert_numpy_types(self, obj):
        """
        Recursively convert numpy types to native Python types for JSON serialization.
        """
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(i) for i in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    
    def _find_non_json_types(self, obj, path="root"):
        """
        Recursively find and report non-JSON-serializable types in nested structures.
        Returns a list of tuples: (path, value, type)
        """
        import json
        issues = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                issues.extend(self._find_non_json_types(value, f"{path}['{key}']"))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                issues.extend(self._find_non_json_types(item, f"{path}[{i}]"))
        else:
            # Try to serialize this specific value
            try:
                json.dumps(obj)
            except (TypeError, ValueError) as e:
                issues.append((path, obj, type(obj).__name__, str(e)))
        
        return issues

    def to_dict(self):
        """
        Convert the config to a dictionary with only JSON-serializable types.
        Assumes stim_list is a pandas DataFrame.
        """
        result = {
            "id": self.id,
            "stim_list": self.stim_list.to_dict(orient="list") if hasattr(self.stim_list, 'to_dict') else self.stim_list,
            "isi": float(self.isi),
            "min_seq_length": int(self.min_seq_length),
            "seq_count": int(self.seq_count) if self.seq_count is not None else None,
            "b_aborted": bool(self.b_aborted),
            "n_rep": int(self.n_rep),
            "max_trials": int(self.max_trials),
            "b_manual_select": bool(self.b_manual_select),
            "b_last_subscr": bool(self.b_last_subscr),
            "aborted_seq": int(self.aborted_seq),
            "aborted_trial": int(self.aborted_trial),
            "seq_length": int(self.seq_length) if self.seq_length is not None else None,
            "sequences": self._convert_numpy_types(self.sequences),
        }
        
        # Include dynamically added attributes if they exist
        if hasattr(self, 'daq_events'):
            result['daq_events'] = self._convert_numpy_types(self.daq_events)
        if hasattr(self, 'photo_times'):
            result['photo_times'] = self._convert_numpy_types(self.photo_times)
        if hasattr(self, 'daq_pic_events'):
            result['daq_pic_events'] = self._convert_numpy_types(self.daq_pic_events)
        if hasattr(self, 'daq_gamepad_events'):
            result['daq_gamepad_events'] = self._convert_numpy_types(self.daq_gamepad_events)
        if hasattr(self, 'line_changes'):
            result['line_changes'] = self._convert_numpy_types(self.line_changes)
        if hasattr(self, 'line_changes_seq_start'):
            result['line_changes_seq_start'] = self._convert_numpy_types(self.line_changes_seq_start)
            
        return result

    @classmethod
    def from_dict(cls, data):
        """
        Create a SubscreeningConfig instance from a dictionary.
        """
        import pandas as pd
        
        # Create a minimal instance with required parameters
        instance = cls.__new__(cls)  # Create without calling __init__
        
        # Set the basic attributes
        instance.scr_id = data["id"]
        instance.id = f'subscreening_{data["id"]}'
        instance.task_name = data.get("task_name", "unknown")
        
        # Convert stim_list back to DataFrame if it was serialized as dict
        stim_list_data = data["stim_list"]
        if isinstance(stim_list_data, dict):
            instance.stim_list = pd.DataFrame(stim_list_data)
        else:
            instance.stim_list = stim_list_data
            
        instance.isi = float(data["isi"])
        instance.min_seq_length = int(data["min_seq_length"])
        instance.seq_count = int(data["seq_count"]) if data["seq_count"] is not None else None
        instance.b_aborted = bool(data["b_aborted"])
        instance.n_rep = int(data["n_rep"])
        instance.max_trials = int(data["max_trials"])
        instance.b_manual_select = bool(data["b_manual_select"])
        instance.b_last_subscr = bool(data["b_last_subscr"])
        instance.aborted_seq = int(data["aborted_seq"])
        instance.aborted_trial = int(data["aborted_trial"])
        instance.seq_length = int(data["seq_length"]) if data["seq_length"] is not None else None
        instance.sequences = data["sequences"]
        
        # Set default values for attributes that might not be in saved data
        instance.sort_n_rep = data.get("sort_n_rep", 1)
        instance.sort_seq_id = data.get("sort_seq_id", 0)
        instance.last_valid_seq = data.get("last_valid_seq", -1)
        instance.stim_on_times = data.get("stim_on_times", [])
        
        # Restore dynamically added attributes if they exist in the data
        if "daq_events" in data:
            instance.daq_events = data["daq_events"]
        if "photo_times" in data:
            instance.photo_times = data["photo_times"]
        if "daq_pic_events" in data:
            instance.daq_pic_events = data["daq_pic_events"]
        if "daq_gamepad_events" in data:
            instance.daq_gamepad_events = data["daq_gamepad_events"]
        if "line_changes" in data:
            instance.line_changes = data["line_changes"]
        if "line_changes_seq_start" in data:
            instance.line_changes_seq_start = data["line_changes_seq_start"]
            
        return instance