import os
import model
from . import record, messaging


class UserCommand(object):
    def __init__(self, username, message, dest_port, model_output):
        self.username = username
        self.message = message
        self.dest_port = destination_port
        self.model_output = model_output

    def save(self):
        """Save the command to the database (Redis)"""
        pass

    @classmethod
    def db_conn(cls):
        """Get Redis connection object to modify database accordingly"""
        pass

    @classmethod
    def get_all(cls, username):
        """Get all saved commands matching the provided username"""
        pass

    @classmethod
    def find_closest_match(cls, username, speech_input):
        """Pass speech_input through the model, get a prediction (a list of
        words) and output the guessed commands (words)

        Returns:
            message
            dest_port
        """
        pass

def register():
    """Go through the register process (ask for the fields in UserCommand) to
    save it to the database.
    """
    print('Registering command...')

    # Prompt for parameters
    username = raw_input('>>> username: ')
    message = raw_input('>>> message: ')
    dest_port = raw_input('>>> destination port: ')

    record_flag = raw_input('>>> press "y" to start recording: ')
    if record_flag == 'y':
        recording, wavfile = record.record_input(wavfile=os.environ['TMP_RECORDING'])

        mfccs = model.utils.wavfile_to_mfccs(wavfile)[0]
        model_output = model.predict(mfccs)

        UserCommand(username, message, dest_port, model_output).save()

def parse():
    """Go through the process of parsing a user's speech (take his username
    + prompt to record), and then feed the recording to the model in order
    to get an output.
    """
    username = raw_input('>>> username: ')
    record_flag = raw_input('>>> press "y" to start recording: ')

    if record_flag == 'y':
        recording, wavfile = record.record_input(wavfile=os.environ['TMP_RECORDING'])

        mfccs = model.utils.wavfile_to_mfccs(wavfile)[0]
        model_output = model.predict(mfccs)

        message, dest_port = UserCommand.find_closest_match(username, model_output)
        messaging.send(message, dest_port)

        print('>>> sending to %d: %s' % (dest_port, message))
