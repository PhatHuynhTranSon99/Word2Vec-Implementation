from scripts.data import process_line

def test_process_line():
    line1 = "Hello World! This is the   man"
    line2 = "His name is John's"
    line3 = "The king is not dead. But you will be  "

    print(process_line(line1))
    assert process_line(line1) == ["hello", "world", "this", "is", "the",  "man"]

    print(process_line(line2))
    assert process_line(line2) == ["his", "name", "is", "johns"]

    print(process_line(line3))
    assert process_line(line3) == ["the", "king", "is", "not", "dead", "but", "you", "will", "be"]