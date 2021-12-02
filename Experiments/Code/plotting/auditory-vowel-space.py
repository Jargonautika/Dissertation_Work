#!/usr/bin/env python3

import joblib

def getVals(exp, experiment):

    matrix_labels = ['Avg. Word Dur.',                                          #* Global == speaker-level word duration mean; Local == utterance-level word duration mean
                     'Avg. Sil. Dur.',                                          #* Global == speaker-level silence duration mean; Local == utterance-level silence duration mean
                     'Dynamic Range',                                           # Global == speaker-level dynamic range; Local == utterance-level dynamic range
                     'Energy',                                                  # Global == speaker-level energy; Local == utterance-level energy
                     'Intensity',                                               # Global == speaker-level intensity; Local == utterance-level intensity
                     'ZCR',                                                     # Global == speaker-level; Local == utterance-level
                     'Root Mean Square',                                        # Global == speaker-level; Local == utterance-level
                     'Sound Pressure Level',                                    # Global == speaker-level; Local == utterance-level
                     'Consonant Vowel Ratio',                                   #* Global == speaker-level global count; Local == utterance-level 
                     'B_CoG', 'B_Kur', 'B_Ske', 'B_Std',                        #* Global == speaker-level global; Local == utterance-level
                     'CH_CoG', 'CH_Kur', 'CH_Ske', 'CH_Std',                    #* ""
                     'D_CoG', 'D_Kur', 'D_Ske', 'D_Std', 
                     'DH_CoG', 'DH_Kur', 'DH_Ske', 'DH_Std', 
                     'CX_CoG', 'CX_Kur', 'CX_Ske', 'CX_Std',                    # NOTE - not present in corpus
                     'EL_CoG', 'EL_Kur', 'EL_Ske', 'EL_Std',                    # NOTE - not present in corpus
                     'EM_CoG', 'EM_Kur', 'EM_Ske', 'EM_Std',                    # NOTE - not present in corpus
                     'EN_CoG', 'EN_Kur', 'EN_Ske', 'EN_Std',                    # NOTE - not present in corpus
                     'F_CoG', 'F_Kur', 'F_Ske', 'F_Std', 
                     'G_CoG', 'G_Kur', 'G_Ske', 'G_Std', 
                     'H_CoG', 'H_Kur', 'H_Ske', 'H_Std', 
                     'JH_CoG', 'JH_Kur', 'JH_Ske', 'JH_Std', 
                     'K_CoG', 'K_Kur', 'K_Ske', 'K_Std', 
                     'L_CoG', 'L_Kur', 'L_Ske', 'L_Std', 
                     'M_CoG', 'M_Kur', 'M_Ske', 'M_Std', 
                     'N_CoG', 'N_Kur', 'N_Ske', 'N_Std', 
                     'NX_CoG', 'NX_Kur', 'NX_Ske', 'NX_Std',                    # NOTE - not present in corpus
                     'NG_CoG', 'NG_Kur', 'NG_Ske', 'NG_Std', 
                     'P_CoG', 'P_Kur', 'P_Ske', 'P_Std', 
                     'Q_CoG', 'Q_Kur', 'Q_Ske', 'Q_Std',                        # NOTE - not present in corpus
                     'R_CoG', 'R_Kur', 'R_Ske', 'R_Std', 
                     'S_CoG', 'S_Kur', 'S_Ske', 'S_Std', 
                     'SH_CoG', 'SH_Kur', 'SH_Ske', 'SH_Std', 
                     'T_CoG', 'T_Kur', 'T_Ske', 'T_Std', 
                     'TH_CoG', 'TH_Kur', 'TH_Ske', 'TH_Std', 
                     'V_CoG', 'V_Kur', 'V_Ske', 'V_Std', 
                     'W_CoG', 'W_Kur', 'W_Ske', 'W_Std', 
                     'WH_CoG', 'WH_Kur', 'WH_Ske', 'WH_Std',                    # NOTE - not present in corpus
                     'Y_CoG', 'Y_Kur', 'Y_Ske', 'Y_Std', 
                     'Z_CoG', 'Z_Kur', 'Z_Ske', 'Z_Std', 
                     'ZH_CoG','ZH_Kur', 'ZH_Ske', 'ZH_Std', 
                     'Avg. Cons. Dur.',                                         #* Global == float: mean of ALL consonants produced by speaker; Local == float: mean of all consonants produced by a speaker during a given utterance
                     'AA-F0_1', 'AA-F0_2', 'AA-F0_3', 'AA-F0_4', 'AA-F0_5',     #* Global == speaker-dependent global measure of AA-F0 at time-stamp 1, etc.; Local == utterance-level
                     'AA-F1_1', 'AA-F2_1', 'AA-F3_1',                           #* Global == speaker-dependent global measure of AA-F1 at time-stamp 1, etc.; Local == utterance-level
                     'AA-F1_2', 'AA-F2_2', 'AA-F3_2',                           #* ""
                     'AA-F1_3', 'AA-F2_3', 'AA-F3_3', 
                     'AA-F1_4', 'AA-F2_4', 'AA-F3_4', 
                     'AA-F1_5', 'AA-F2_5', 'AA-F3_5', 
                     'AE-F0_1', 'AE-F0_2', 'AE-F0_3', 'AE-F0_4', 'AE-F0_5', 
                     'AE-F1_1', 'AE-F2_1', 'AE-F3_1', 
                     'AE-F1_2', 'AE-F2_2', 'AE-F3_2', 
                     'AE-F1_3', 'AE-F2_3', 'AE-F3_3', 
                     'AE-F1_4', 'AE-F2_4', 'AE-F3_4', 
                     'AE-F1_5', 'AE-F2_5', 'AE-F3_5', 
                     'AH-F0_1', 'AH-F0_2', 'AH-F0_3', 'AH-F0_4', 'AH-F0_5', 
                     'AH-F1_1', 'AH-F2_1', 'AH-F3_1', 
                     'AH-F1_2', 'AH-F2_2', 'AH-F3_2', 
                     'AH-F1_3', 'AH-F2_3', 'AH-F3_3', 
                     'AH-F1_4', 'AH-F2_4', 'AH-F3_4', 
                     'AH-F1_5', 'AH-F2_5', 'AH-F3_5', 
                     'AO-F0_1', 'AO-F0_2', 'AO-F0_3', 'AO-F0_4', 'AO-F0_5', 
                     'AO-F1_1', 'AO-F2_1', 'AO-F3_1', 
                     'AO-F1_2', 'AO-F2_2', 'AO-F3_2', 
                     'AO-F1_3', 'AO-F2_3', 'AO-F3_3', 
                     'AO-F1_4', 'AO-F2_4', 'AO-F3_4', 
                     'AO-F1_5', 'AO-F2_5', 'AO-F3_5', 
                     'AW-F0_1', 'AW-F0_2', 'AW-F0_3', 'AW-F0_4', 'AW-F0_5', 
                     'AW-F1_1', 'AW-F2_1', 'AW-F3_1', 
                     'AW-F1_2', 'AW-F2_2', 'AW-F3_2', 
                     'AW-F1_3', 'AW-F2_3', 'AW-F3_3', 
                     'AW-F1_4', 'AW-F2_4', 'AW-F3_4', 
                     'AW-F1_5', 'AW-F2_5', 'AW-F3_5', 
                     'AX-F0_1', 'AX-F0_2', 'AX-F0_3', 'AX-F0_4', 'AX-F0_5',     # NOTE - not present in corpus
                     'AX-F1_1', 'AX-F2_1', 'AX-F3_1',
                     'AX-F1_2', 'AX-F2_2', 'AX-F3_2', 
                     'AX-F1_3', 'AX-F2_3', 'AX-F3_3', 
                     'AX-F1_4', 'AX-F2_4', 'AX-F3_4', 
                     'AX-F1_5', 'AX-F2_5', 'AX-F3_5', 
                     'AXR-F0_1', 'AXR-F0_2', 'AXR-F0_3', 'AXR-F0_4', 'AXR-F0_5',# NOTE - not present in corpus
                     'AXR-F1_1', 'AXR-F2_1', 'AXR-F3_1', 
                     'AXR-F1_2', 'AXR-F2_2', 'AXR-F3_2', 
                     'AXR-F1_3', 'AXR-F2_3', 'AXR-F3_3', 
                     'AXR-F1_4', 'AXR-F2_4', 'AXR-F3_4', 
                     'AXR-F1_5', 'AXR-F2_5', 'AXR-F3_5', 
                     'AY-F0_1', 'AY-F0_2', 'AY-F0_3', 'AY-F0_4', 'AY-F0_5', 
                     'AY-F1_1', 'AY-F2_1', 'AY-F3_1', 
                     'AY-F1_2', 'AY-F2_2', 'AY-F3_2', 
                     'AY-F1_3', 'AY-F2_3', 'AY-F3_3', 
                     'AY-F1_4', 'AY-F2_4', 'AY-F3_4', 
                     'AY-F1_5', 'AY-F2_5', 'AY-F3_5', 
                     'EH-F0_1', 'EH-F0_2', 'EH-F0_3', 'EH-F0_4', 'EH-F0_5', 
                     'EH-F1_1', 'EH-F2_1', 'EH-F3_1', 
                     'EH-F1_2', 'EH-F2_2', 'EH-F3_2', 
                     'EH-F1_3', 'EH-F2_3', 'EH-F3_3', 
                     'EH-F1_4', 'EH-F2_4', 'EH-F3_4', 
                     'EH-F1_5', 'EH-F2_5', 'EH-F3_5', 
                     'ER-F0_1', 'ER-F0_2', 'ER-F0_3', 'ER-F0_4', 'ER-F0_5', 
                     'ER-F1_1', 'ER-F2_1', 'ER-F3_1', 
                     'ER-F1_2', 'ER-F2_2', 'ER-F3_2', 
                     'ER-F1_3', 'ER-F2_3', 'ER-F3_3', 
                     'ER-F1_4', 'ER-F2_4', 'ER-F3_4', 
                     'ER-F1_5', 'ER-F2_5', 'ER-F3_5', 
                     'EY-F0_1', 'EY-F0_2', 'EY-F0_3', 'EY-F0_4', 'EY-F0_5', 
                     'EY-F1_1', 'EY-F2_1', 'EY-F3_1', 
                     'EY-F1_2', 'EY-F2_2', 'EY-F3_2', 
                     'EY-F1_3', 'EY-F2_3', 'EY-F3_3', 
                     'EY-F1_4', 'EY-F2_4', 'EY-F3_4', 
                     'EY-F1_5', 'EY-F2_5', 'EY-F3_5', 
                     'IH-F0_1', 'IH-F0_2', 'IH-F0_3', 'IH-F0_4', 'IH-F0_5', 
                     'IH-F1_1', 'IH-F2_1', 'IH-F3_1', 
                     'IH-F1_2', 'IH-F2_2', 'IH-F3_2', 
                     'IH-F1_3', 'IH-F2_3', 'IH-F3_3', 
                     'IH-F1_4', 'IH-F2_4', 'IH-F3_4', 
                     'IH-F1_5', 'IH-F2_5', 'IH-F3_5', 
                     'IX-F0_1', 'IX-F0_2', 'IX-F0_3', 'IX-F0_4', 'IX-F0_5',     # NOTE - not present in corpus
                     'IX-F1_1', 'IX-F2_1', 'IX-F3_1', 
                     'IX-F1_2', 'IX-F2_2', 'IX-F3_2', 
                     'IX-F1_3', 'IX-F2_3', 'IX-F3_3', 
                     'IX-F1_4', 'IX-F2_4', 'IX-F3_4', 
                     'IX-F1_5', 'IX-F2_5', 'IX-F3_5', 
                     'IY-F0_1', 'IY-F0_2', 'IY-F0_3', 'IY-F0_4', 'IY-F0_5', 
                     'IY-F1_1', 'IY-F2_1', 'IY-F3_1', 
                     'IY-F1_2', 'IY-F2_2', 'IY-F3_2', 
                     'IY-F1_3', 'IY-F2_3', 'IY-F3_3', 
                     'IY-F1_4', 'IY-F2_4', 'IY-F3_4', 
                     'IY-F1_5', 'IY-F2_5', 'IY-F3_5', 
                     'OW-F0_1', 'OW-F0_2', 'OW-F0_3', 'OW-F0_4', 'OW-F0_5', 
                     'OW-F1_1', 'OW-F2_1', 'OW-F3_1', 
                     'OW-F1_2', 'OW-F2_2', 'OW-F3_2', 
                     'OW-F1_3', 'OW-F2_3', 'OW-F3_3', 
                     'OW-F1_4', 'OW-F2_4', 'OW-F3_4', 
                     'OW-F1_5', 'OW-F2_5', 'OW-F3_5', 
                     'OY-F0_1', 'OY-F0_2', 'OY-F0_3', 'OY-F0_4', 'OY-F0_5', 
                     'OY-F1_1', 'OY-F2_1', 'OY-F3_1', 
                     'OY-F1_2', 'OY-F2_2', 'OY-F3_2', 
                     'OY-F1_3', 'OY-F2_3', 'OY-F3_3', 
                     'OY-F1_4', 'OY-F2_4', 'OY-F3_4', 
                     'OY-F1_5', 'OY-F2_5', 'OY-F3_5', 
                     'UH-F0_1', 'UH-F0_2', 'UH-F0_3', 'UH-F0_4', 'UH-F0_5', 
                     'UH-F1_1', 'UH-F2_1', 'UH-F3_1', 
                     'UH-F1_2', 'UH-F2_2', 'UH-F3_2', 
                     'UH-F1_3', 'UH-F2_3', 'UH-F3_3', 
                     'UH-F1_4', 'UH-F2_4', 'UH-F3_4', 
                     'UH-F1_5', 'UH-F2_5', 'UH-F3_5', 
                     'UW-F0_1', 'UW-F0_2', 'UW-F0_3', 'UW-F0_4', 'UW-F0_5', 
                     'UW-F1_1', 'UW-F2_1', 'UW-F3_1', 
                     'UW-F1_2', 'UW-F2_2', 'UW-F3_2', 
                     'UW-F1_3', 'UW-F2_3', 'UW-F3_3', 
                     'UW-F1_4', 'UW-F2_4', 'UW-F3_4', 
                     'UW-F1_5', 'UW-F2_5', 'UW-F3_5', 
                     'UX-F0_1', 'UX-F0_2', 'UX-F0_3', 'UX-F0_4', 'UX-F0_5',     # NOTE - not present in corpus
                     'UX-F1_1', 'UX-F2_1', 'UX-F3_1', 
                     'UX-F1_2', 'UX-F2_2', 'UX-F3_2', 
                     'UX-F1_3', 'UX-F2_3', 'UX-F3_3', 
                     'UX-F1_4', 'UX-F2_4', 'UX-F3_4', 
                     'UX-F1_5', 'UX-F2_5', 'UX-F3_5', 
                     'Avg. Voca. Dur.',                                         #* Global == float: mean of ALL consonants produced by speaker; Local == utterance-level
                     'DWGP-SMN',                                                # TODO Global == utterance-level; Local == utterance-level
                     'DWGP-SSN']                                                # TODO Global == utterance-level; Local == utterance-level

    y_train = joblib.load('{}/vectors/classifiers/{}-y_train.pkl'.format(exp, experiment))
    X_train = joblib.load('{}/vectors/classifiers/{}-X_train.pkl'.format(exp, experiment))
    trainSpeakerDict = joblib.load('{}/vectors/classifiers/{}-trainSpeakerDict.pkl'.format(exp, experiment))

    y_test = joblib.load('{}/vectors/classifiers/{}-y_test.pkl'.format(exp, experiment))
    X_test = joblib.load('{}/vectors/classifiers/{}-X_test.pkl'.format(exp, experiment))
    devSpeakerDict = joblib.load('{}/vectors/classifiers/{}-devSpeakerDict.pkl'.format(exp, experiment))
    
    scaler = joblib.load('{}/vectors/classifiers/{}-scaler.pkl'.format(exp, experiment))

    print(1)

    Speaker A - 
        UTT01 AAx3
        UTT02 AAx4
        UTT03 AAx5

    Global - 
            np.mean(AA)

    518 features

    [
    ['SPKR-A', '0.2']
    ['SPKR-A', '0.2']
    ['SPKR-A', '0.2']
    ]



def main():

    myList = [('auditory-global-loso', '/tmp/tmp.TcRZJvM59O'),
              ('auditory-local-loso', '/tmp/tmp.OgrVUR6iE4'),
              ('auditory-global-withheld', '/tmp/tmp.7NmQXNGAif'),
              ('auditory-local-withheld', '/tmp/tmp.IHHR7Bl2yh')]

    for name, ddir in myList:

        getVals(ddir, name.rsplit('-', 1)[0])

if __name__ == "__main__":

    main()