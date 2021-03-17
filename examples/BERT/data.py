import glob
import os
import torch
import logging
from torchtext.data.utils import get_tokenizer
import random
from torchtext.experimental.datasets import LanguageModelingDataset
from torch.utils.data.datasets import ListDirFilesIterableDataset, LoadFilesFromDiskIterableDataset


###################################################################
# Set up dataset for book corpus
###################################################################
def BookCorpus(vocab, tokenizer=get_tokenizer("basic_english"),
               data_select=('train', 'valid', 'test'), removed_tokens=[],
               min_sentence_len=None):

    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset(set(('train', 'test', 'valid'))):
        raise TypeError('data_select is not supported!')

    extracted_files = glob.glob('/datasets01/bookcorpus/021819/*/*.txt')
    random.seed(1000)
    random.shuffle(extracted_files)

    num_files = len(extracted_files)
    _path = {'train': extracted_files[:(num_files // 20 * 17)],
             'test': extracted_files[(num_files // 20 * 17):(num_files // 20 * 18)],
             'valid': extracted_files[(num_files // 20 * 18):]}

    data = {}
    for item in _path.keys():
        data[item] = []
        logging.info('Creating {} data'.format(item))
        tokens = []
        for txt_file in _path[item]:
            with open(txt_file, 'r', encoding="utf8", errors='ignore') as f:
                for line in f.readlines():
                    _tokens = tokenizer(line.strip())
                    if min_sentence_len:
                        if len(_tokens) >= min_sentence_len:
                            tokens.append([vocab.stoi[token] for token in _tokens])
                    else:
                        tokens += [vocab.stoi[token] for token in _tokens]
        data[item] = tokens

    for key in data_select:
        if data[key] == []:
            raise TypeError('Dataset {} is empty!'.format(key))
    if min_sentence_len:
        return tuple(LanguageModelingDataset(data[d], vocab, lambda x: x, False)
                     for d in data_select)
    else:
        return tuple(LanguageModelingDataset(torch.tensor(data[d]).long(), vocab, lambda x: x, False)
                     for d in data_select)


class CC100(torch.utils.data.IterableDataset):
    def __init__(self, data_directory, languages, start_line=0, num_lines=16, reset_iterator=False, shuffle=False):
        """
        Args:
            data_directory: directory to save the data files
            languages: a unix style filter string set for filtering filename
            start_line: the starting line, if None, it will start at random line
            num_lines: the number of output lines
            reset_iterator: reset file handle and continuously read text until the number of lines
            shuffle: flag to shuffle the datasets by languages. Default: False

        Examples:
            >>> from data import CC100
            >>> dataset = CC100('/datasets01/cc100/031720/', {'as_IN.txt', 'om_KE.txt', 'su_ID.txt'}, start_line=300, num_lines=10)
            >>> for rec in dataset:
            >>>     print(rec)
        """

        file_paths = ListDirFilesIterableDataset(data_directory, languages)
        self.dataset_list = [item for item in LoadFilesFromDiskIterableDataset(file_paths)]
        if shuffle:
            random.shuffle(self.dataset_list)
        self.start_line = start_line
        self.num_lines = num_lines
        self.datasets_begin_end = {}
        for (_file_path, _buffer) in self.dataset_list:
            _filename = os.path.split(_file_path)[-1]
            max_line = CC100_FILES_LINES[_filename]
            if self.start_line is not None:
                _begin = self.start_line
            else:
                _begin = random.randint(0, max(max_line - num_lines, 0))
            self.datasets_begin_end[_filename] = (_begin, min(_begin + num_lines, max_line))
        self.reset_iterator = reset_iterator
        self._count = 0
        self._current_dataset = 0

    def __iter__(self):
        for i, (filename, dataset_handle) in enumerate(self.dataset_list):
            language_id = self.find_language_id(filename)
            filename = os.path.split(filename)[-1]
            start_line, end_line = self.datasets_begin_end[filename]
            self.setup_dataset(start_line, dataset_handle)
            for _count in range(end_line - start_line):
                _text = self.readline(dataset_handle)
                if not _text:
                    if self.reset_iterator:
                        dataset_handle.seek(0)
                        _text = self.readline(dataset_handle)
                    else:
                        continue
                yield language_id, _text.decode('utf-8')

    def __len__(self):
        total_lines = 0
        for (_file_path, _buffer) in self.dataset_list:
            filename = os.path.split(_file_path)[-1]
            start_line, end_line = self.datasets_begin_end[filename]
            total_lines += end_line - start_line
        return total_lines

    def setup_dataset(self, start_line, dataset_handle):
        for _count in range(start_line):
            _text = self.readline(dataset_handle)

    # readline skips newline
    def readline(self, dataset_handle):
        _text = dataset_handle.readline()
        while _text == b'\n':
            _text = dataset_handle.readline()
        return _text

    def find_language_id(self, file_name):
        for idx in range(len(language_list)):
            if language_list[idx] in file_name:
                return idx
        return -1


language_list = ['af_ZA', 'am_ET', 'ar_AR', 'as_IN', 'az_AZ',
                 'be_BY', 'bg_BG', 'bn_IN', 'bn_IN_rom', 'br_FR',
                 'bs_BA', 'ca_ES', 'cs_CZ', 'cy_GB', 'da_DK',
                 'de_DE', 'el_GR', 'en_XX', 'eo_EO', 'es_XX',
                 'et_EE', 'eu_ES', 'fa_IR', 'ff_NG', 'fi_FI',
                 'fr_XX', 'fy_NL', 'ga_IE', 'gd_GB', 'gl_ES',
                 'gn_PY', 'gu_IN', 'ha_NG', 'he_IL', 'hi_IN',
                 'hi_IN_rom', 'hr_HR', 'ht_HT', 'hu_HU', 'hy_AM',
                 'id_ID', 'ig_NG', 'is_IS', 'it_IT', 'ja_XX',
                 'jv_ID', 'ka_GE', 'kk_KZ', 'km_KH', 'kn_IN',
                 'ko_KR', 'ku_TR', 'ky_KG', 'la_VA', 'lg_UG',
                 'li_NL', 'ln_CD', 'lo_LA', 'lt_LT', 'lv_LV',
                 'mg_MG', 'mk_MK', 'ml_IN', 'mn_MN', 'mr_IN',
                 'ms_MY', 'my_MM', 'my_MM_zaw', 'ne_NP', 'nl_XX',
                 'no_XX', 'ns_ZA', 'om_KE', 'or_IN', 'pa_IN',
                 'pl_PL', 'ps_AF', 'pt_XX', 'qu_PE', 'rm_CH',
                 'ro_RO', 'ru_RU', 'sa_IN', 'sc_IT', 'sd_PK',
                 'si_LK', 'sk_SK', 'sl_SI', 'so_SO', 'sq_AL',
                 'sr_RS', 'ss_SZ', 'su_ID', 'sv_SE', 'sw_KE',
                 'ta_IN', 'ta_IN_rom', 'te_IN', 'te_IN_rom', 'th_TH',
                 'tl_XX', 'tn_BW', 'tr_TR', 'ug_CN', 'uk_UA',
                 'ur_PK', 'ur_PK_rom', 'uz_UZ', 'vi_VN', 'wo_SN',
                 'xh_ZA', 'yi_DE', 'yo_NG', 'zh_CN', 'zh_TW']

CC100_FILES_LINES = {
    'af_ZA.txt': 9973471, 'am_ET.txt': 3124561, 'ar_AR.txt': 126199432, 'as_IN.txt': 490379, 'az_AZ.txt': 41379968, 'be_BY.txt': 18474129,
    'bg_BG.txt': 235459726, 'bn_IN.txt': 57929082, 'bn_IN_rom.txt': 18638681, 'br_FR.txt': 1122957, 'bs_BA.txt': 1015534,
    'ca_ES.txt': 77707813, 'cs_CZ.txt': 124277022, 'cy_GB.txt': 7767408, 'da_DK.txt': 343692674, 'de_DE.txt': 417071103,
    'el_GR.txt': 200954802, 'en_XX.txt': 2105324624, 'eo_EO.txt': 7612069, 'es_XX.txt': 379272696, 'et_EE.txt': 45953570,
    'eu_ES.txt': 16761238, 'fa_IR.txt': 611039245, 'ff_NG.txt': 87667, 'fi_FI.txt': 376523863, 'fr_XX.txt': 427630359,
    'fy_NL.txt': 1575804, 'ga_IE.txt': 4118906, 'gd_GB.txt': 1111419, 'gl_ES.txt': 15479749, 'gn_PY.txt': 72285, 'gu_IN.txt': 9046487,
    'ha_NG.txt': 3655296, 'he_IL.txt': 207542919, 'hi_IN.txt': 103537752, 'hi_IN_rom.txt': 10251114, 'hr_HR.txt': 143627479,
    'ht_HT.txt': 418169, 'hu_HU.txt': 384533104, 'hy_AM.txt': 25355483, 'id_ID.txt': 1048306652, 'ig_NG.txt': 365023,
    'is_IS.txt': 25400262, 'it_IT.txt': 178838153, 'ja_XX.txt': 458387942, 'jv_ID.txt': 1416729, 'ka_GE.txt': 31708119,
    'kk_KZ.txt': 35587324, 'km_KH.txt': 7190193, 'kn_IN.txt': 13519328, 'ko_KR.txt': 390127563, 'ku_TR.txt': 2719375,
    'ky_KG.txt': 5193248, 'la_VA.txt': 18105390, 'lg_UG.txt': 406579, 'li_NL.txt': 76047, 'ln_CD.txt': 143425,
    'lo_LA.txt': 2773595, 'lt_LT.txt': 111143031, 'lv_LV.txt': 68418444, 'mg_MG.txt': 1310498, 'mk_MK.txt': 20209317,
    'ml_IN.txt': 24822142, 'mn_MN.txt': 15098167, 'mr_IN.txt': 12232071, 'ms_MY.txt': 77522478, 'my_MM.txt': 2207994,
    'my_MM_zaw.txt': 11516761, 'ne_NP.txt': 12732810, 'nl_XX.txt': 229914131, 'no_XX.txt': 338137974, 'ns_ZA.txt': 129546,
    'om_KE.txt': 440213, 'or_IN.txt': 2182352, 'pa_IN.txt': 4147866, 'pl_PL.txt': 256234282, 'ps_AF.txt': 4532180,
    'pt_XX.txt': 339889917, 'qu_PE.txt': 112851, 'rm_CH.txt': 167166, 'ro_RO.txt': 391186642, 'ru_RU.txt': 848845934,
    'sa_IN.txt': 2049884, 'sc_IT.txt': 7527, 'sd_PK.txt': 1396852, 'si_LK.txt': 12643262, 'sk_SK.txt': 174236545,
    'sl_SI.txt': 74653572, 'so_SO.txt': 2266581, 'sq_AL.txt': 35969391, 'sr_RS.txt': 35747957, 'ss_SZ.txt': 4440, 'su_ID.txt': 387322,
    'sv_SE.txt': 580387314, 'sw_KE.txt': 12660806,
    'ta_IN.txt': 68237343, 'ta_IN_rom.txt': 6243679, 'te_IN.txt': 17484093, 'te_IN_rom.txt': 6867186, 'th_TH.txt': 318507542,
    'tl_XX.txt': 29600482, 'tn_BW.txt': 747103, 'tr_TR.txt': 127698803, 'ug_CN.txt': 1564913, 'uk_UA.txt': 357405323,
    'ur_PK.txt': 27951610, 'ur_PK_rom.txt': 14901617, 'uz_UZ.txt': 5176271, 'vi_VN.txt': 991671222, 'wo_SN.txt': 347198,
    'xh_ZA.txt': 776651, 'yi_DE.txt': 1778592, 'yo_NG.txt': 76533, 'zh_CN.txt': 208732858, 'zh_TW.txt': 85165683,
}
