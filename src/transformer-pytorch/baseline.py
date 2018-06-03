with open("baseline/pred") as f, open("baseline/pred.sen", 'w') as g:
    for line in f:
        try:
            line = line[0:line.index(" </s>")]
        except ValueError:
            pass
        print(line, file= g)
