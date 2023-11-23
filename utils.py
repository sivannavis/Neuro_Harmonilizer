class Note:
    def __init__(self, note_name):
        self.name = note_name


class NoteSeq:
    def __init__(self, start_note):
        self.start = start_note

    def get_current_circle(self):
        # return clock wise circle of fifth that begins with current start note
        notes = ['C', 'G', 'D', 'A', 'E', 'B', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F']
        circle = []