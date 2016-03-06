"""
allow user to record, make a prediction and call the necessary function in the
processor (just call functions from all modules)

rough process:
- Record_user_command(): # wait until the user says something (while silence:
    don’t record) and, once the user has spoken for X seconds, record the signal
- Interpret the user command
- Let the processor know that that command was called
"""
