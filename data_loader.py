"""
data_loader.py
MIDI 파일을 파싱하여 멜로디 데이터(음정·offset·악기)를 추출하고,
전처리된 결과를 pickle로 저장/로드한다.
"""

import os
import pickle
import logging

import music21

from config import PREPROCESSED_DIR, PREDICT_PREPROCESSED_DIR

logger = logging.getLogger(__name__)


def _store_chord(mel_data, track_idx, chord, inst):
    offset = chord.offset
    if offset not in mel_data[track_idx]:
        mel_data[track_idx][offset] = {}
    mel_data[track_idx][offset]['offset'] = offset
    for p in chord.pitches:
        mel_data[track_idx][offset]['note'] = p.midi
    mel_data[track_idx][offset]['instrument'] = inst


def _store_note(mel_data, track_idx, note, inst):
    offset = note.offset
    if offset not in mel_data[track_idx]:
        mel_data[track_idx][offset] = {}
    mel_data[track_idx][offset]['offset'] = offset
    prev_p = 0
    for p in note.pitches:
        if prev_p < p.midi:
            mel_data[track_idx][offset]['note'] = p.midi
        prev_p = p.midi
    mel_data[track_idx][offset]['instrument'] = inst


def create_mel_data_each_file(midi_obj):
    """MIDI 파일 하나에서 트랙별 멜로디 데이터(음정·offset·악기)를 추출한다."""
    c = midi_obj.flat.getElementsByClass(music21.instrument.Instrument)
    mel_data = []

    for i, m in enumerate(midi_obj):
        inst = None
        mel_data.append(dict())

        for n in m:
            if n in c:
                inst = n.midiProgram

            if isinstance(n, music21.stream.Voice):
                for x in n:
                    if x in c:
                        inst = x.midiProgram
                    if isinstance(x, music21.chord.Chord):
                        _store_chord(mel_data, i, x, inst)
                    elif isinstance(x, music21.note.Note):
                        _store_note(mel_data, i, x, inst)
            elif isinstance(n, music21.chord.Chord):
                _store_chord(mel_data, i, n, inst)
            elif isinstance(n, music21.note.Note):
                _store_note(mel_data, i, n, inst)

    total_notes = sum(len(a) for a in mel_data)
    logger.info("추출된 노트 수: %d", total_notes)
    return mel_data


def get_midi_set(data_path, path_name, predict=False):
    """디렉토리의 MIDI 파일을 파싱하고 결과를 pickle로 저장한 뒤 반환한다."""
    file_list = [
        os.path.join(data_path, f)
        for f in os.listdir(data_path)
        if f.endswith('.mid') or f.endswith('.midi')
    ]

    mel_arr_list = []
    for file_name in file_list:
        logger.info("처리 중: %s", file_name)
        try:
            midi_obj = music21.converter.parse(file_name)
            mel_data = create_mel_data_each_file(midi_obj)
        except Exception as e:
            logger.warning("파일 처리 실패 (%s): %s", file_name, e)
            continue

        mel_arr = []
        for mel_data_i in mel_data:
            for key in sorted(mel_data_i):
                mel_arr.append(mel_data_i[key])
        mel_arr_list.append(mel_arr)

    preprocessed_dir = PREDICT_PREPROCESSED_DIR if predict else PREPROCESSED_DIR
    os.makedirs(preprocessed_dir, exist_ok=True)

    save_path = os.path.join(preprocessed_dir, f"{path_name}_mel_arr_list.p")
    with open(save_path, "wb") as fp:
        pickle.dump(mel_arr_list, fp)
    logger.info("저장 완료: %s", save_path)

    return mel_arr_list
