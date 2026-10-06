from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

DEFAULT_LANGUAGE = "english"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_SAMPLER_DECODE_STEPS = 1
DEFAULT_NOISE_CLAMP = None
DEFAULT_EOS_THRESHOLD = -4.0
DEFAULT_FRAMES_AFTER_EOS = None
# TODO: make this dynamic since english_2026-04 supports bigger chunks
MAX_TOKEN_PER_CHUNK = 50

DEFAULT_TEXT_FOR_LANGUAGE = {
    "english": (
        "Hello world. I am Kyutai's Pocket TTS. "
        "I'm fast enough to run on small CPUs. "
        "I hope you'll like me."
    ),
    "french": (
        "Bonjour le monde. Je suis le TTS de poche de Kyutai. "
        "Je suis assez rapide pour fonctionner sur de petits CPU. "
        "J'espère que vous m'aimerez."
    ),
    "german": (
        "Hallo Welt. Ich bin Pocket TTS von Kyutai. "
        "Ich bin schnell genug, um auch auf kleinen CPUs zu laufen. "
        "Ich hoffe, ich gefalle dir."
    ),
    "portuguese": (
        "Olá mundo. Eu sou o Pocket TTS da Kyutai. "
        "Sou rápido o suficiente para rodar em CPUs pequenas. "
        "Espero que você goste de mim."
    ),
    "italian": (
        "Ciao mondo. Sono il Pocket TTS di Kyutai. "
        "Sono abbastanza veloce da funzionare su piccole CPU. "
        "Spero che ti piacerò."
    ),
    "dutch": (
        "Hallo wereld. Ik ben Pocket TTS van Kyutai. "
        "Ik ben snel genoeg om op kleine CPU's te draaien. "
        "Ik hoop dat je me leuk vindt."
    ),
    "spanish": (
        "Hola mundo. Soy el Pocket TTS de Kyutai. "
        "Soy lo suficientemente rápido para funcionar en pequeñas CPU. "
        "Espero que te guste."
    ),
    "indonesian": (
        "Halo dunia. Saya Pocket TTS dari Kyutai. "
        "Saya cukup cepat untuk berjalan di CPU kecil. "
        "Saya harap kamu menyukai saya."
    ),
    "korean": (
        "안녕하세요 세계. 저는 키유타의 포켓 TTS입니다. "
        "작은 CPU에서도 실행할 수 있을 만큼 빠릅니다. "
        "저를 좋아해 주시면 좋겠어요."
    ),
    "czech": (
        "Ahoj světe. Jsem Pocket TTS od Kyutai. "
        "Jsem dost rychlý na to, abych běžel i na malých procesorech. "
        "Doufám, že se vám budu líbit."
    ),
    "greek": (
        "Γεια σου κόσμε. Είμαι το Pocket TTS της Kyutai. "
        "Είμαι αρκετά γρήγορο ώστε να τρέχω σε μικρούς επεξεργαστές. "
        "Ελπίζω να σας αρέσω."
    ),
    "polish": (
        "Witaj świecie. Jestem Pocket TTS od Kyutai. "
        "Jestem na tyle szybki, że działam na małych procesorach. "
        "Mam nadzieję, że mnie polubisz."
    ),
    "russian": (
        "Привет, мир. Я Pocket TTS от Kyutai. "
        "Я достаточно быстр, чтобы работать на маленьких процессорах. "
        "Надеюсь, я вам понравлюсь."
    ),
    "hindi": (
        "नमस्ते दुनिया। मैं Kyutai का Pocket TTS हूँ। "
        "मैं इतना तेज़ हूँ कि छोटे CPU पर भी चल सकता हूँ। "
        "मुझे उम्मीद है कि आपको मैं पसंद आऊँगा।"
    ),
    "estonian": (
        "Tere, maailm. Ma olen Kyutai Pocket TTS. "
        "Ma olen piisavalt kiire, et töötada väikestel protsessoritel. "
        "Loodan, et sulle meeldin."
    ),
    "turkish": (
        "Merhaba dünya. Ben Kyutai'nin Pocket TTS'iyim. "
        "Küçük işlemcilerde çalışacak kadar hızlıyım. "
        "Umarım beni beğenirsiniz."
    ),
}

DEFAULT_VOICE_FOR_LANGUAGE = {
    "italian": "giovanni",
    "spanish": "lola",
    "german": "juergen",
    "portuguese": "rafael",
    "french": "estelle",
    "dutch": "daan",
}
DEFAULT_VOICE_FALLBACK = "alba"
# Predefined voices are states precomputed with the released weights of a language model,
# so neither a custom config nor a training checkpoint can use them. For those we default
# to the audio file behind the fallback voice: any model can clone it.
DEFAULT_VOICE_FOR_CUSTOM_MODEL = _ORIGINS_OF_PREDEFINED_VOICES[DEFAULT_VOICE_FALLBACK]


def get_default_text_for_language(language: str | None) -> str:
    for key, text in DEFAULT_TEXT_FOR_LANGUAGE.items():
        if language is not None and key in language:
            return text
    return DEFAULT_TEXT_FOR_LANGUAGE[DEFAULT_LANGUAGE]


def get_default_voice_for_language(
    language: str | None, config: str | None = None, checkpoint: str | None = None
) -> str:
    """The voice to use when the user didn't pick one.

    `config` and `checkpoint` both mean custom weights, which cannot use the predefined
    voices, hence the audio file instead of the voice name.
    """
    if config is not None or checkpoint is not None:
        return DEFAULT_VOICE_FOR_CUSTOM_MODEL
    for key, voice in DEFAULT_VOICE_FOR_LANGUAGE.items():
        if language is not None and key in language:
            return voice
    return DEFAULT_VOICE_FALLBACK
