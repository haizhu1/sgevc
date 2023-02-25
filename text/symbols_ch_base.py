""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
_pad        = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
# Export all symbols:
#symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
symbols = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH', 'a1', 'a2', 'a3', 'a4', 'a5', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'an1', 'an2', 'an3', 'an4', 'an5', 'ang1', 'ang2', 'ang3', 'ang4', 'ang5', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'b', 'c', 'ch', 'd', 'e1', 'e2', 'e3', 'e4', 'e5', 'ei1', 'ei2', 'ei3', 'ei4', 'ei5', 'en1', 'en2', 'en3', 'en4', 'en5', 'eng1', 'eng2', 'eng3', 'eng4', 'eng5', 'er2', 'er3', 'er4', 'er5', 'f', 'g', 'h', 'i1', 'i2', 'i3', 'i4', 'i5', 'ia1', 'ia2', 'ia3', 'ia4', 'ia5', 'ian1', 'ian2', 'ian3', 'ian4', 'ian5', 'iang1', 'iang2', 'iang3', 'iang4', 'iang5', 'iao1', 'iao2', 'iao3', 'iao4', 'iao5', 'ie1', 'ie2', 'ie3', 'ie4', 'ie5', 'ii1', 'ii2', 'ii3', 'ii4', 'ii5', 'iii1', 'iii2', 'iii3', 'iii4', 'iii5', 'in1', 'in2', 'in3', 'in4', 'in5', 'ing1', 'ing2', 'ing3', 'ing4', 'ing5', 'io1', 'io2', 'io4', 'io5', 'iong1', 'iong2', 'iong3', 'iong4', 'iong5', 'iou1', 'iou2', 'iou3', 'iou4', 'iou5', 'j', 'k', 'l', 'm', 'n', 'o1', 'o2', 'o3', 'o4', 'o5', 'ong1', 'ong2', 'ong3', 'ong4', 'ong5', 'ou1', 'ou2', 'ou3', 'ou4', 'ou5', 'p', 'pp0', 'pp1', 'pp2', 'pp3', 'pp4', 'q', 'r', 's', 'sh', 'sil', 't', 'u1', 'u2', 'u3', 'u4', 'u5', 'ua1', 'ua2', 'ua3', 'ua4', 'ua5', 'uai1', 'uai2', 'uai3', 'uai4', 'uai5', 'uan1', 'uan2', 'uan3', 'uan4', 'uan5', 'uang1', 'uang2', 'uang3', 'uang4', 'uang5', 'uei1', 'uei2', 'uei3', 'uei4', 'uei5', 'uen1', 'uen2', 'uen3', 'uen4', 'uen5', 'ueng1', 'ueng3', 'ueng4', 'uo1', 'uo2', 'uo3', 'uo4', 'uo5', 'v1', 'v2', 'v3', 'v4', 'v5', 'van1', 'van2', 'van3', 'van4', 'van5', 've1', 've2', 've3', 've4', 'vn1', 'vn2', 'vn3', 'vn4', 'x', 'z', 'zh'] 
# Special symbol ids
#SPACE_ID = symbols.index(" ")
