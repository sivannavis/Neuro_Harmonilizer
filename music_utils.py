class Note:
    def __init__(self, note_name):
        self.name = note_name
        self.next = None
        self.last = None

    def add_next(self, note):
        self.next = note

    def add_last(self, note):
        self.last = note


def get_circle(notes):
    circle = []
    # one time forward pass to add last and next note
    note = Note(notes[0])
    circle.append(note)
    for n in range(1, len(notes)-1):
        note = Note(notes[n])
        note.add_last(circle[n-1])
        circle[n-1].add_next(note)
        circle.append(note)
    circle[0].add_last(circle[-1])
    return circle


class NoteSeq:
    def __init__(self, start_note):
        self.start = start_note

    def get_circle_seq(self):
        # return clock wise circle of fifth that begins with current start note
        notes = ['C', 'G', 'D', 'A', 'E', 'B', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F']
        circle = get_circle(notes)
        return circle

    def get_norm_seq(self):
        notes = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        circle = get_circle(notes)
        return circle


    def search_note(self, orientation):
        pass

    def search_fifth(self, note1, note2):
        pass



if __name__ == '__main__':
    circle_5 = NoteSeq('B')
