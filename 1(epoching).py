import numpy as np
import pickle
from pathlib import Path
from utils.config import DATA_PROCESSED_DIR, DATA_EPOCHS_DIR
import mne


DATA_DIR = Path(DATA_PROCESSED_DIR).parent  
EPOCHS_DIR = DATA_DIR / "epochs"
EPOCHS_DIR.mkdir(parents=True, exist_ok=True)  

# subject search
subject_dirs = sorted([d for d in Path(DATA_PROCESSED_DIR).iterdir() if d.is_dir() and d.name.startswith('S')])

if not subject_dirs:
    raise FileNotFoundError(f" No subject directories found in {DATA_PROCESSED_DIR}")

print(f" Found {len(subject_dirs)} subject directories: {[d.name for d in subject_dirs]}")

# event id mapping
event_mapping = {
    "T0": 1,
    "T1": 2,
    "T2": 3,
    "T0_T1": 4,
    "T1_T0": 5,
    "T0_T2": 6,
    "T2_T0": 7
}

# Epoching
tmin, tmax = -2.0, 2.0  # cue onset -2s~2s
baseline = (None, 0)  # Baseline correction

for subj_dir in subject_dirs:
    eeg_files = list(subj_dir.glob("*.pkl"))
    if not eeg_files:
        print(f" No processed EEG files found in {subj_dir}")
        continue

    print(f"\n Processing subject: {subj_dir.name} ({len(eeg_files)} files)")

    for eeg_file in eeg_files:
        try:
            print(f"  File: {eeg_file.name}...")

           
            with open(eeg_file, "rb") as f:
                data = pickle.load(f)

            raw = data["raw"]

           
            annotations = raw.annotations
            event_desc = annotations.description
            event_times = (annotations.onset * raw.info['sfreq']).astype(int)

          
            print(f"     Events: {event_desc}")

            if len(event_desc) < 2:
                print(f" Not enough events.")
                continue

            # event array 
            events = np.column_stack([event_times, np.zeros(len(event_times), dtype=int),
                                      [event_mapping[e] for e in event_desc]])

            # removing repeated event
            unique_times, unique_indices = np.unique(events[:, 0], return_index=True)
            events = events[unique_indices]

            # event change 
            transition_events = []
            for i in range(len(events) - 1):
                cur_event_time, _, cur_event_id = events[i]
                next_event_time, _, next_event_id = events[i + 1]

                if cur_event_id == event_mapping["T0"] and next_event_id == event_mapping["T1"]:
                    transition_events.append([next_event_time, 0, event_mapping["T0_T1"]])
                elif cur_event_id == event_mapping["T1"] and next_event_id == event_mapping["T0"]:
                    transition_events.append([next_event_time, 0, event_mapping["T1_T0"]])
                elif cur_event_id == event_mapping["T0"] and next_event_id == event_mapping["T2"]:
                    transition_events.append([next_event_time, 0, event_mapping["T0_T2"]])
                elif cur_event_id == event_mapping["T2"] and next_event_id == event_mapping["T0"]:
                    transition_events.append([next_event_time, 0, event_mapping["T2_T0"]])

            
            if transition_events:
                events = np.vstack([events, transition_events])
                events = events[np.argsort(events[:, 0])]

            if len(events) == 0:
                print(f"   No valid events")
                continue

            
            picks = mne.pick_channels(
                raw.info["ch_names"],
                include=['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..']
            )
            if len(picks) == 0:
                print(f" No valid EEG channels")
                continue

            # epoch
            epochs = mne.Epochs(
                raw, events, event_id=event_mapping,
                tmin=tmin, tmax=tmax, baseline=baseline,
                picks=picks, preload=True,
                event_repeated='drop'
            )

            
            subj_epochs_dir = EPOCHS_DIR / subj_dir.name
            subj_epochs_dir.mkdir(parents=True, exist_ok=True)
            save_path = subj_epochs_dir / f"{eeg_file.stem}_epochs.pkl"

            with open(save_path, "wb") as f:
                pickle.dump({"epochs": epochs}, f)

            if save_path.exists():
                print(f"  Saved: {save_path}")
            else:
                print(f"   Failed to save: {save_path}")

      

print("\n All subjects/epochs processed successfully!")

