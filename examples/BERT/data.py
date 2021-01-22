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
    def __init__(self, data_directory, languages, start_line=0, chunk=16):
        """

        Examples:
            >>> from data import CC100
            >>> dataset = CC100('/datasets01/cc100/031720/', {'as_IN.txt', 'om_KE.txt', 'su_ID.txt'}, start_line=300, chunk=10)
            >>> for rec in dataset:
            >>>     print(rec)
        """

        file_paths = ListDirFilesIterableDataset(data_directory, languages)
        self.dataset_list = [item for item in LoadFilesFromDiskIterableDataset(file_paths)]
        self.start_line = start_line
        self.chunk = chunk
        self._count = 0
        self._current_dataset = 0

    def __iter__(self):
        for i, (filename, dataset_handle) in enumerate(self.dataset_list):
            language_id = self.find_language_id(filename)
            self.setup_dataset(dataset_handle)
            for _count in range(self.chunk):
                _text = self.readline(dataset_handle)
                yield language_id, _text.decode('utf-8')

    def setup_dataset(self, dataset_handle):
        for _count in range(self.start_line):
            _text = self.readline(dataset_handle)

    def readline(self, dataset_handle):
        _text = dataset_handle.readline()
        if not _text:
            dataset_handle.seek(0)
            _text = dataset_handle.readline()
        while _text == b'\n':
            _text = dataset_handle.readline()
            if not _text:
                dataset_handle.seek(0)
                _text = dataset_handle.readline()
        return _text

    def find_language_id(self, file_name):
        for idx in range(len(language_list)):
            if language_list[idx] in file_name:
                return idx
        return -1


language_list = ['af_ZA', 'cs_CZ', 'fi_FI', 'hr_HR', 'km_KH', 'mg_MG', 'om_KE', 'sd_PK', 'ta_IN_rom', 'vi_VN',
                 'am_ET', 'cy_GB', 'fr_XX', 'ht_HT', 'kn_IN', 'mk_MK', 'or_IN', 'si_LK', 'te_IN', 'wo_SN',
                 'ar_AR', 'da_DK', 'fy_NL', 'hu_HU', 'ko_KR', 'ml_IN', 'pa_IN', 'sk_SK', 'te_IN_rom', 'xh_ZA',
                 'as_IN', 'de_DE', 'ga_IE', 'hy_AM', 'ku_TR', 'mn_MN', 'pl_PL', 'sl_SI', 'th_TH', 'yi_DE',
                 'az_AZ', 'el_GR', 'gd_GB', 'id_ID', 'ky_KG', 'mr_IN', 'ps_AF', 'so_SO', 'tl_XX', 'yo_NG',
                 'be_BY', 'en_XX', 'gl_ES', 'ig_NG', 'la_VA', 'ms_MY', 'pt_XX', 'sq_AL', 'tn_BW', 'zh_CN',
                 'bg_BG', 'eo_EO', 'gn_PY', 'is_IS', 'lg_UG', 'my_MM', 'qu_PE', 'sr_RS', 'tr_TR', 'zh_TW',
                 'bn_IN', 'es_XX', 'gu_IN', 'it_IT', 'li_NL', 'my_MM_zaw', 'rm_CH', 'ss_SZ', 'ug_CN',
                 'bn_IN_rom', 'et_EE', 'ha_NG', 'ja_XX', 'ln_CD', 'ne_NP', 'ro_RO', 'su_ID', 'uk_UA',
                 'br_FR', 'eu_ES', 'he_IL', 'jv_ID', 'lo_LA', 'nl_XX', 'ru_RU', 'sv_SE', 'ur_PK',
                 'bs_BA', 'fa_IR', 'hi_IN', 'ka_GE', 'lt_LT', 'no_XX', 'sa_IN', 'sw_KE', 'ur_PK_rom',
                 'ca_ES', 'ff_NG', 'hi_IN_rom', 'kk_KZ', 'lv_LV', 'ns_ZA', 'sc_IT', 'ta_IN', 'uz_UZ']
