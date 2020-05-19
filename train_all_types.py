import json
from train_utils import parse_agrs, train

if __name__ == "__main__":

    hparams = parse_agrs()
    to_write = []
    i = 0
    for thresholding_algorithm in ['hard', 'soft']:
        if thresholding_algorithm == 'hard':
            hparams.thresholding_parameter = 0.2
        else:
            hparams.thresholding_parameter = 0.01
        for threshold_mode in ['level_dependent', 'global']:
            for trainable_wavelets in [True, False]:
                for trainable_threshold in [True, False]:
                    if trainable_wavelets:
                        for wavelet_name in [None, 'db4']:
                            hparams.thresholding_algorithm = thresholding_algorithm
                            hparams.threshold_mode = threshold_mode
                            hparams.trainable_wavelets = trainable_wavelets
                            hparams.trainable_threshold = trainable_threshold
                            hparams.wavelet_name = wavelet_name

                            _, mean_rmse, mean_snr = train(hparams)

                            trial = [thresholding_algorithm, threshold_mode, trainable_wavelets,
                                           trainable_threshold, wavelet_name, mean_rmse, mean_snr]
                            print(trial)

                            to_write.append(trial)
                            with open('Logs.txt', 'w') as filehandle:
                                json.dump(to_write, filehandle)
                            i += 1
                    else:
                        wavelet_name = 'db4'
                        hparams.thresholding_algorithm = thresholding_algorithm
                        hparams.threshold_mode = threshold_mode
                        hparams.trainable_wavelets = trainable_wavelets
                        hparams.trainable_threshold = trainable_threshold
                        hparams.wavelet_name = wavelet_name

                        if trainable_threshold:
                            _, mean_rmse, mean_snr = train(hparams)
                        else:
                            _, mean_rmse, mean_snr = train(hparams, evaluate_only=True)

                        trial = [thresholding_algorithm, threshold_mode, trainable_wavelets,
                                       trainable_threshold, wavelet_name, mean_rmse, mean_snr]
                        print(trial)
                        to_write.append(trial)
                        with open('Logs.txt', 'w') as filehandle:
                            json.dump(to_write, filehandle)
                        i += 1


    with open('Logs.txt', 'w') as filehandle:
        json.dump(to_write, filehandle)



