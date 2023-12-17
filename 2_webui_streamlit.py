import os
import logging
import re_matching
from tools.sentence import split_by_language
import torch
import utils
from infer import infer, latest_version, get_net_g, infer_multilang
import numpy as np
from config import config
from tools.translate import translate
import librosa
import streamlit as st
import gradio as gr

# 알림을 최소화
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

net_g = None

device = config.webui_config.device
if device == "mps":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 오디오를 생성
def generate_audio(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
    reference_audio,
    emotion,
    skip_start=False,
    skip_end=False,
):
    audio_list = []
    with torch.no_grad():
        for idx, piece in enumerate(slices):
            skip_start = (idx != 0) and skip_start
            skip_end = (idx != len(slices) - 1) and skip_end
            audio = infer(
                piece,
                reference_audio=reference_audio,
                emotion=emotion,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language,
                hps=hps,
                net_g=net_g,
                device=device,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
    return audio_list

# 오디오를 2개 이상의 언어로 생성. language의 dropdown에서 mix를 선택했을 경우에만 사용되야 함
def generate_audio_multilang(
    slices,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    speaker,
    language,
    reference_audio,
    emotion,
    skip_start=False,
    skip_end=False,
):
    audio_list = []
    with torch.no_grad():
        for idx, piece in enumerate(slices):
            skip_start = (idx != 0) and skip_start
            skip_end = (idx != len(slices) - 1) and skip_end
            audio = infer_multilang(
                piece,
                reference_audio=reference_audio,
                emotion=emotion,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language[idx],
                hps=hps,
                net_g=net_g,
                device=device,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
    return audio_list

# 나뉜 텍스트들을 다른 정보와 취합하여 오디오로 변환하여 출력
def tts_split(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    cut_by_sent,
    interval_between_para,
    interval_between_sent,
    reference_audio,
    emotion,
):
    if language == "mix":
        return ("invalid", None)
    while text.find("\n\n") != -1:
        text = text.replace("\n\n", "\n")
    para_list = re_matching.cut_para(text)
    audio_list = []
    if not cut_by_sent:
        for idx, p in enumerate(para_list):
            skip_start = idx != 0
            skip_end = idx != len(para_list) - 1
            audio = infer(
                p,
                reference_audio=reference_audio,
                emotion=emotion,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
                sid=speaker,
                language=language,
                hps=hps,
                net_g=net_g,
                device=device,
                skip_start=skip_start,
                skip_end=skip_end,
            )
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
            audio_list.append(audio16bit)
            silence = np.zeros((int)(44100 * interval_between_para), dtype=np.int16)
            #audio_list.append(silence)
    else:
        for idx, p in enumerate(para_list):
            skip_start = idx != 0
            skip_end = idx != len(para_list) - 1
            audio_list_sent = []
            sent_list = re_matching.cut_sent(p)
            for idx, s in enumerate(sent_list):
                skip_start = (idx != 0) and skip_start
                skip_end = (idx != len(sent_list) - 1) and skip_end
                audio = infer(
                    s,
                    reference_audio=reference_audio,
                    emotion=emotion,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise_scale,
                    noise_scale_w=noise_scale_w,
                    length_scale=length_scale,
                    sid=speaker,
                    language=language,
                    hps=hps,
                    net_g=net_g,
                    device=device,
                    skip_start=skip_start,
                    skip_end=skip_end,
                )
                audio_list_sent.append(audio)
                silence = np.zeros((int)(44100 * interval_between_sent))
                audio_list_sent.append(silence)
            if (interval_between_para - interval_between_sent) > 0:
                silence = np.zeros(
                    (int)(44100 * (interval_between_para - interval_between_sent))
                )
                audio_list_sent.append(silence)
            audio16bit = gr.processing_utils.convert_to_16_bit_wav(
                np.concatenate(audio_list_sent)
            )  # 对完整句子做音量归一
            audio_list.append(audio16bit)
    audio_concat = np.concatenate(audio_list)
    return ("Success", (44100, audio_concat))


def tts_fn(
    text: str,
    speaker,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    language,
    reference_audio,
    emotion,
    prompt_mode,
):
    if prompt_mode == "Audio prompt":
        if reference_audio == None:
            return ("Invalid audio prompt", None)
        else:
            reference_audio = load_audio(reference_audio)[1]
    else:
        reference_audio = None
    audio_list = []
    if language == "mix":
        bool_valid, str_valid = re_matching.validate_text(text)
        if not bool_valid:
            return str_valid, (
                hps.data.sampling_rate,
                np.concatenatenp.concatenate([np.zeros(hps.data.sampling_rate // 2)]),
            )
        result = []
        for slice in re_matching.text_matching(text):
            _speaker = slice.pop()
            temp_contant = []
            temp_lang = []
            for lang, content in slice:
                if "|" in content:
                    temp = []
                    temp_ = []
                    for i in content.split("|"):
                        if i != "":
                            temp.append([i])
                            temp_.append([lang])
                        else:
                            temp.append([])
                            temp_.append([])
                    temp_contant += temp
                    temp_lang += temp_
                else:
                    if len(temp_contant) == 0:
                        temp_contant.append([])
                        temp_lang.append([])
                    temp_contant[-1].append(content)
                    temp_lang[-1].append(lang)
            for i, j in zip(temp_lang, temp_contant):
                result.append([*zip(i, j), _speaker])
        for i, one in enumerate(result):
            skip_start = i != 0
            skip_end = i != len(result) - 1
            _speaker = one.pop()
            idx = 0
            while idx < len(one):
                text_to_generate = []
                lang_to_generate = []
                while True:
                    lang, content = one[idx]
                    temp_text = [content]
                    if len(text_to_generate) > 0:
                        text_to_generate[-1] += [temp_text.pop(0)]
                        lang_to_generate[-1] += [lang]
                    if len(temp_text) > 0:
                        text_to_generate += [[i] for i in temp_text]
                        lang_to_generate += [[lang]] * len(temp_text)
                    if idx + 1 < len(one):
                        idx += 1
                    else:
                        break
                skip_start = (idx != 0) and skip_start
                skip_end = (idx != len(one) - 1) and skip_end
                print(text_to_generate, lang_to_generate)
                audio_list.extend(
                    generate_audio_multilang(
                        text_to_generate,
                        sdp_ratio,
                        noise_scale,
                        noise_scale_w,
                        length_scale,
                        _speaker,
                        lang_to_generate,
                        reference_audio,
                        emotion,
                        skip_start,
                        skip_end,
                    )
                )
                idx += 1
    elif language.lower() == "auto":
        for idx, slice in enumerate(text.split("|")):
            if slice == "":
                continue
            skip_start = idx != 0
            skip_end = idx != len(text.split("|")) - 1
            sentences_list = split_by_language(slice, target_languages=["zh", "ja", "en"])
            idx = 0
            while idx < len(sentences_list):
                text_to_generate = []
                lang_to_generate = []
                while True:
                    content, lang = sentences_list[idx]
                    temp_text = [content]
                    lang = lang.upper()
                    if lang == "JA":
                        lang = "JP"
                    if len(text_to_generate) > 0:
                        text_to_generate[-1] += [temp_text.pop(0)]
                        lang_to_generate[-1] += [lang]
                    if len(temp_text) > 0:
                        text_to_generate += [[i] for i in temp_text]
                        lang_to_generate += [[lang]] * len(temp_text)
                    if idx + 1 < len(sentences_list):
                        idx += 1
                    else:
                        break
                skip_start = (idx != 0) and skip_start
                skip_end = (idx != len(sentences_list) - 1) and skip_end
                print(text_to_generate, lang_to_generate)
                audio_list.extend(
                    generate_audio_multilang(
                        text_to_generate,
                        sdp_ratio,
                        noise_scale,
                        noise_scale_w,
                        length_scale,
                        speaker,
                        lang_to_generate,
                        reference_audio,
                        emotion,
                        skip_start,
                        skip_end,
                    )
                )
                idx += 1
    else:
        audio_list.extend(
            generate_audio(
                text.split("|"),
                sdp_ratio,
                noise_scale,
                noise_scale_w,
                length_scale,
                speaker,
                language,
                reference_audio,
                emotion,
            )
        )

    audio_concat = np.concatenate(audio_list)
    return "Success", (hps.data.sampling_rate, audio_concat)


def load_audio(path):
    audio, sr = librosa.load(path, 48000)
    return sr, audio


def gr_util(item):
    if item == "Text prompt":
        return {"visible": True, "__type__": "update"}, {
            "visible": False,
            "__type__": "update",
        }
    else:
        return {"visible": False, "__type__": "update"}, {
            "visible": True,
            "__type__": "update",
        }


if __name__ == "__main__":
    if config.webui_config.debug:
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)
    hps = utils.get_hparams_from_file(config.webui_config.config_path)
    version = hps.version if hasattr(hps, "version") else latest_version
    net_g = get_net_g(
        model_path=config.webui_config.model, version=version, device=device, hps=hps
    )
    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    languages = ["ZH", "JP", "EN", "mix", "auto"]

    st.title("Your Streamlit App Title")

    text = st.text_area(
        "Input Text",
        """
        If you choose the language as 'mix', you must enter the format, otherwise an error will occur:
            Format example (zh is Chinese, jp is Japanese, case insensitive; speaker example: gongzi):
             [Speaker1]<zh>Hello, こんにちは! <jp>こんにちは、世界。
             [Speaker2]<zh>How are you?<jp>元気ですか？
             [Speaker3]<zh>Thank you.<jp>どういたしまして。
             ...
        Additionally, all language options can split long paragraphs into sentences using '|'.
        """,
    )

    trans = st.button("Translate (中翻日)")
    slicer = st.button("Quick Slice")

    speaker = st.selectbox("Speaker", speakers)
    prompt_mode = st.radio("Prompt Mode", ["Text prompt", "Audio prompt"])

    text_prompt = st.text_input("Text prompt", "Happy")
    audio_prompt = st.file_uploader("Audio prompt", type=["wav", "mp3"], key="audio_prompt")
    sdp_ratio = st.slider("SDP Ratio", 0.0, 1.0, 0.2, 0.1)
    noise_scale = st.slider("Noise", 0.1, 2.0, 0.6, 0.1)
    noise_scale_w = st.slider("Noise_W", 0.1, 2.0, 0.8, 0.1)
    length_scale = st.slider("Length", 0.1, 2.0, 1.0, 0.1)
    language = st.selectbox("Language", languages)

    btn = st.button("Generate Audio!")

    st.markdown(
        "Prompt Mode: Choose between 'Text prompt' or 'Audio prompt'. Use 'Text prompt' to describe the desired style in text (e.g., Happy). Use 'Audio prompt' to upload an audio file for style reference."
    )

    st.markdown("Parameters:")
    interval_between_sent = st.slider(
        "Sentence Interval (seconds)", 0.0, 5.0, 0.2, 0.1
    )
    interval_between_para = st.slider(
        "Paragraph Interval (seconds)",
        0.0,
        10.0,
        1.0,
        0.1,
        help="Should be greater than Sentence Interval to take effect.",
    )
    opt_cut_by_sent = st.checkbox(
        "Split by Sentence",
        help="Split the text into sentences in addition to splitting by paragraphs.",
    )

    st.markdown("**Output:**")
    text_output = st.text_area("Status Information", value="", key="text_output")
    #audio_output = st.audio("Output Audio", format="audio/wav", key="audio_output")
    #audio_output = st.audio("Output Audio", format="audio/wav")
    audio_output = st.empty()

    if btn:
        text_output, audio_concat = tts_fn(
            text,
            speaker,
            sdp_ratio,
            noise_scale,
            noise_scale_w,
            length_scale,
            language,
            reference_audio=audio_prompt,
            emotion=text_prompt,
            prompt_mode=prompt_mode,
        )
        if text_output == "Success":
            sample_rate, audio_data = audio_concat
            audio_output.audio(audio_data.astype(np.int16), format="audio/wav", sample_rate=sample_rate)
            #audio_output= st.audio(audio_concat, format="audio/wav")

    trans_button_clicked = False
    slicer_button_clicked = False

    if trans:
        text = translate(text)
        trans_button_clicked = True

    if slicer:
        text_output, audio_concat = tts_split(
            text,
            speaker,
            sdp_ratio,
            noise_scale,
            noise_scale_w,
            length_scale,
            language,
            opt_cut_by_sent,
            interval_between_para,
            interval_between_sent,
            reference_audio=audio_prompt,
            emotion=text_prompt,
        )
        slicer_button_clicked = True

    if trans_button_clicked:
        st.success("Translation completed.")
    elif slicer_button_clicked:
        st.success("Slicing and generation completed.")

    if prompt_mode == "Audio prompt":
        st.markdown(
            "Note: If 'Audio prompt' is selected, the reference audio should be provided in the 'Audio prompt' section above."
        )

    st.markdown("Inference page is now active!")
