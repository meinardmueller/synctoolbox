import numpy as np
import pandas as pd
from synctoolbox.feature.pitch import __visualize_pitch
from synctoolbox.feature.pitch_onset import __f_peaks_to_matrix
import matplotlib.pyplot as plt


def read_csv_to_df(csv_filepath: str = '',
                   csv_delimiter: str = ';') -> pd.DataFrame:
    """Reads .csv file containing symbolic music into a pandas DataFrame.
    Column names are normalized to be lower case.

    Parameters
    ----------
    csv_filepath : str
        Filepath to the .csv file.

    csv_delimiter : str
        Delimiter of the .csv file (default: ';')

    Returns
    -------
    df : pd.Dataframe
        Annotations in pandas Dataframe format.
    """

    df = pd.read_csv(filepath_or_buffer=csv_filepath,
                     delimiter=csv_delimiter)
    df.columns = df.columns.str.lower()

    if 'pitch' in df.columns:
        df['pitch'] = df['pitch'].astype(int)

    return df


def df_to_pitch_features(df: pd.DataFrame,
                         feature_rate: float,
                         midi_min: int = 21,
                         midi_max: int = 108,
                         transpose: int = 0,
                         ignore_velocity: bool = False,
                         ignore_percussion: bool = False,
                         visualize: bool = False,
                         visualization_title: str = "Pitch features") -> np.ndarray:
    """ Computes pitch-based features for a dataframe containing symbolic music.
    The resulting features have the same format as the output of 'audio_to_pitch_features'
    for audio.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of symbolic music piece. Can be loaded with 'read_csv_to_df'.
        WARNING! Column names are supposed to be lowercase.

    feature_rate: float
        Desired features per second of the output representation

    midi_min : int
        Minimum MIDI index (indices below ``midi_min`` are filled with zero in the output)

    midi_max : int
        Maximum MIDI index (indices above ``midi_max`` are filled with zero in the output)

    transpose : int
        Number of semi-tones the symbolic information should be transposed

    ignore_velocity : bool
        If `True`, STMSP values aren't set based on MIDI velocities, just
        uses 0 and 1.

    ignore_percussion : bool
        Ignores percussion. If `True`, no features are generated for percussive events.

    visualize : bool
        Set `True` to activate the visualization of features

    visualization_title : str
        Title for the visualization plot. Only relevant if 'visualize' is True

    Returns
    -------
    f_pitch : np.ndarray
        Matrix of size 128xN containing the extracted pitch-based features
    """
    stepsize_ms = 1 / feature_rate * 1000
    audio_duration = __get_audio_duration_from_df(df)  # in seconds
    num_pitch_features = np.ceil(audio_duration * 1000 / stepsize_ms).astype(int)
    f_pitch = np.zeros((128, num_pitch_features), dtype=np.float64)

    for _, row in df.iterrows():
        start_time_ms = 1000 * row['start']
        end_time_ms = 1000 * (row['start'] + row['duration'])
        pitch = int(row['pitch'] + transpose)
        velocity = row['velocity']
        instrument = row['instrument']
        first_step_size_interval = np.floor(start_time_ms / stepsize_ms).astype(int) + 1
        last_step_size_interval = np.minimum(np.floor(end_time_ms / stepsize_ms) + 1,
                                             num_pitch_features)
        first_window_involved = first_step_size_interval.astype(int)
        last_window_involved = np.minimum(last_step_size_interval + 1, num_pitch_features).astype(int)

        if not midi_max >= pitch >= midi_min:
            raise ValueError(f'The pitch for note {pitch} at time point {start_time_ms/ 1000} sec is not valid.')

        # TODO: ATTENTION TO INDEXING!
        if instrument == 'percussive':
            # TODO The code for percussive events is not well tested!
            if not ignore_percussion:
                for cur_win in range(first_window_involved, np.minimum(first_window_involved + 1, num_pitch_features) + 1):
                    f_pitch[:, cur_win-1] = __compute_pitch_energy(cur_energy_val=f_pitch[:, cur_win - 1],
                                                                   cur_win=cur_win,
                                                                   start_time_ms=start_time_ms,
                                                                   end_time_ms=end_time_ms,
                                                                   velocity=velocity,
                                                                   is_percussive=True,
                                                                   stepsize_ms=stepsize_ms)
        else:
            for cur_win in range(first_window_involved, last_window_involved + 1):
                f_pitch[pitch, cur_win-1] = __compute_pitch_energy(cur_energy_val=f_pitch[pitch, cur_win - 1],
                                                                   cur_win=cur_win,
                                                                   start_time_ms=start_time_ms,
                                                                   end_time_ms=end_time_ms,
                                                                   velocity=velocity,
                                                                   is_percussive=False,
                                                                   stepsize_ms=stepsize_ms,
                                                                   ignore_velocity=ignore_velocity)

    if visualize:
        __visualize_pitch(f_pitch, feature_rate=feature_rate, plot_title=visualization_title)
        plt.show()

    return f_pitch


def df_to_pitch_onset_features(df: pd.DataFrame,
                               midi_min: int = 21,
                               midi_max: int = 108,
                               transpose: int = 0,
                               ignore_percussion: bool = False,
                               peak_height_scale_factor: float = 1e6,
                               visualize: bool = False,
                               visualization_title: str = "Pitch features") -> dict:
    """Computes pitch-based onset features for a dataframe containing symbolic music.
    The resulting features have the same format as the output of 'audio_to_pitch_onset_features'
    for audio.

    Parameters
    ----------
    df : pd.Dataframe
        Dataframe of symbolic music piece. Can be loaded with 'read_csv_to_df'.
        WARNING! Column names are supposed to be lowercase.

    midi_min : int
        Minimum MIDI index (indices below 'midi_min' will raise an error)

    midi_max : int
        Maximum MIDI index (indices above 'midi_max' will raise an error)

    transpose : int
        Number of semi-tones the midi information should be transposed

    ignore_percussion : bool
        Ignores percussion. If `True`, no features are generated for percussive events.

    peak_height_scale_factor : float
        Scales the midi velocity so the resulting feature values
        are in a similar range than the peak features from an audio file
        So 1e6 is more or less arbitrary.

    visualize : bool
        Set `True` to activate the visualization of features

    visualization_title : str
        Title for the visualization plot. Only relevant if 'visualize' is True

    Returns
    -------
    f_peaks : dict
        A dictionary of onset peaks, see 'audio_to_pitch_onset_features' for the exact format
    """
    peak_height_scale_factor_percussion = peak_height_scale_factor * 10
    num_peaks_in_pitch = {pitch: 0 for pitch in range(1, 129)}
    f_peaks = dict()

    for _, row in df.iterrows():
        start_time_ms = 1000 * row['start']
        pitch = row['pitch'] + transpose
        velocity = row['velocity']
        instrument = row['instrument']

        if not midi_max >= pitch >= midi_min:
            raise ValueError(f'The pitch for note {pitch} at timepoint {start_time_ms/ 1000} sec is not valid.')

        def add_peak_for_pitch(p, scale_factor=peak_height_scale_factor):
            num_peaks_in_pitch[p] += 1

            if num_peaks_in_pitch[p] > 0:
                if p not in f_peaks:
                    f_peaks[p] = np.zeros((2, 1000))
                if num_peaks_in_pitch[p] > f_peaks[p].shape[1]:
                    f_peaks[p] = np.concatenate([f_peaks[p], np.zeros((2, 1000))], axis=1)

            if p not in f_peaks or f_peaks[p].size == 0:
                f_peaks[p] = np.array([[start_time_ms], [velocity / scale_factor]], np.float64)
            else:
                f_peaks[p][0, num_peaks_in_pitch[p] - 1] = start_time_ms
                f_peaks[p][1, num_peaks_in_pitch[p] - 1] = velocity / scale_factor

        if instrument == 'percussive':
            # TODO The code for percussive events is not well tested!
            if not ignore_percussion:
                for p in range(1, 129):
                    add_peak_for_pitch(p, peak_height_scale_factor_percussion)
        else:
            add_peak_for_pitch(pitch)

    for pitch in f_peaks:
        time_vals = f_peaks[pitch][0, :][0: num_peaks_in_pitch[pitch]]
        peak_vals = f_peaks[pitch][1, :][0: num_peaks_in_pitch[pitch]]
        f_peaks[pitch][0, :time_vals.size] = time_vals
        f_peaks[pitch][1, :peak_vals.size] = peak_vals
        sort_index = np.argsort(f_peaks[pitch][0, :])
        f_peaks[pitch][0, :] = f_peaks[pitch][0, :][sort_index]
        f_peaks[pitch][1, :] = f_peaks[pitch][1, :][sort_index]

    if visualize:
        highest_time_res = 50 / 22050  # TODO don't hardcode this, identify appropriate resolution based on df
        imagedata = __f_peaks_to_matrix(np.max(df["start"] + df["duration"]), f_peaks, highest_time_res, midi_max, midi_min)
        __visualize_pitch(imagedata.T, midi_min, midi_max, feature_rate=1 / highest_time_res, plot_title=visualization_title)
        plt.show()

    return f_peaks


def __compute_pitch_energy(cur_energy_val: float,
                           cur_win: int,
                           start_time_ms: float,
                           end_time_ms: float,
                           velocity: float,
                           stepsize_ms: float = 100.0,
                           ignore_velocity: bool = False,
                           is_percussive: bool = False) -> float:
    """TODO Add description

    Parameters
    ----------
    cur_energy_val : float
        Current energy value at the corresponding pitch index

    cur_win : int
        Current window

    start_time_ms : float
        Starting time of the sound event in milliseconds

    end_time_ms : float
        Ending time of the sound event in milliseconds

    velocity : float
        Key velocity

    stepsize_ms : float
        Stepsize of the features in milliseconds

    ignore_velocity : bool
        If `True`, STMSP values aren't set based on MIDI velocities, just
        uses 0 and 1.

    is_percussive : bool
        Set `True`, if the instrument is percussive.

    Returns
    -------
    res : float
        Computed energy value in the corresponding index
    """

    right_border_cur_win_ms = cur_win * stepsize_ms
    left_border_cur_win_ms = right_border_cur_win_ms - 2 * stepsize_ms

    contribution = (np.minimum(end_time_ms, right_border_cur_win_ms) -
                    np.maximum(start_time_ms, left_border_cur_win_ms)) / (2 * stepsize_ms)

    if is_percussive:
        # Add energy equally distributed to all pitches
        # Since we assume the percussive sound to be short,
        # we just add energy to the first relevant Window
        res = cur_energy_val + (velocity / 128) * contribution
    else:
        # If not percussive,
        # add energy for this note to features
        # assume constant note energy throughout the whole note
        #  this may later be improved to an ADSR model
        if ignore_velocity:
            res = 1.0
        else:
            res = cur_energy_val + velocity * contribution

    return res


def __get_audio_duration_from_df(df: pd.DataFrame) -> float:
    """Gets the duration of the symbolic file (end of the last sound event)

    Parameters
    ----------
    df : pd.Dataframe
        Input dataframe having 'start' and 'duration' OR 'end'

    Returns
    -------
    duration : float
        Duration of the audio.
    """
    for column in df.columns:
        if column == 'end':
            duration = df[column].max()
            return duration

    if 'start' not in df.columns or 'duration' not in df.columns:
        raise ValueError('start and duration OR end must be within the columns of the'
                         'dataframe.')

    df['end'] = df['start'] + df['duration']
    duration = df['end'].max()
    return duration

