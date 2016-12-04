import subprocess
def Calculate_Score(prediction,original):
    """Calculate the score depending on R scripts (Score_R.R) file.
    Args:
        * prediction: the predicted ranks.
        * original: the correct values
    Return:
        * single float value
    """
    # Define command and arguments
    command = 'Rscript'
    path2script = './Score_R.R'
    args = [str(x) for x in prediction.flatten().tolist()] + [str(x) for x in original.flatten().tolist()] 
    cmd = [command, path2script]+args

    # check_output will run the command and store to result
    x = subprocess.check_output(cmd, universal_newlines=True)
    return float(x[4:])
