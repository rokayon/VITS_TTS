# we use the same path as this script as our training folder.
output_path = 'content/tts_model'

# audio_config = BaseAudioConfig(
#      sample_rate = 22050,
#     resample =True
# )

audio_config = VitsAudioConfig(
    sample_rate=22050, 
    win_length=1024, 
    hop_length=256, 
    num_mels=80, 
    mel_fmin=0, 
    mel_fmax=None
)


if(male):
    characters_config = CharactersConfig(
    pad = '<PAD>',
    eos = '।', #'<EOS>', #'।',
    bos = '<BOS>',# None,
    blank = '<BLNK>',
    phonemes = None,
    characters =  "তট৫ভিঐঋখঊড়ইজমএেঘঙসীঢ়হঞ‘ঈকণ৬ঁৗশঢঠ‌১্২৮দৃঔগও—ছউংবৈঝাযফ‍চরষঅৌৎথড়৪ধ০ুূ৩আঃপয়’নলো",
    punctuations = "-!,|.? ",
    )
else:
    characters_config = CharactersConfig(
    pad = '<PAD>',
    eos = '।', #'<EOS>', #'।',
    bos = '<BOS>',# None,
    blank = '<BLNK>',
    phonemes = None,
    characters =  "ইগং়’ুঃন১ঝূও‘ঊোছপফৈ৮ষযৎঢঈকঠিজ০৬ীটডএঅঋধচে২৩ণউয়ঢ়খলভৗসহ্ড়দথবঔাঞশরৌম—ঐআৃঘঙ‌ঁ৪৫ত",
    punctuations = ".?-!|, ",
    )


# VitsConfig: all model related values for training, validating and testing.

config = VitsConfig(
    audio=audio_config,
    run_name="vits_4_nov",
    batch_size=8, # Reduced batch size
    eval_batch_size=4, # Reduced eval batch size
    batch_group_size=0,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1,
    text_cleaner=None, #"collapse_whitespace"
    use_phonemes=False,
    compute_input_seq_cache=True,
    print_step=100,
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    characters = characters_config,
    save_step=1000,
    cudnn_benchmark=True,
    test_sentences = [
        'হয়,হয়ে,ওয়া,হয়েছ,হয়েছে,দিয়ে,যায়,দায়,নিশ্চয়,আয়,ভয়,নয়,আয়াত,নিয়ে,হয়েছে,দিয়েছ,রয়ে,রয়েছ,রয়েছে।',
        'দেয়,দেওয়া,বিষয়,হয়,হওয়া,সম্প্রদায়,সময়,হয়েছি,দিয়েছি,হয়,হয়েছিল,বিষয়ে,নয়,কিয়াম,ইয়া,দেয়া,দিয়েছে,আয়াতে,দয়া।',
        'ইয়াহুদ,নয়,ব্যয়,ইয়াহুদী,নেওয়া,উভয়ে,যায়,হয়েছিল,প্রয়োজন।'
    ]
)
