import datetime

start_time = datetime.datetime.now()

for _ in range(100000):
    dictx = {'a': 1, 'b': 2, 'c': 3}
    inverted_dict = {}
    for key in dictx:
        inverted_dict[dictx[key]] = key

end_time = datetime.datetime.now()

print(inverted_dict)
print(end_time-start_time)