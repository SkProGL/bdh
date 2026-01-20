def make_counting_data(
    max_number=100,
    window=10,
    lines=1000
):
    data = []
    for i in range(lines):
        start = i % (max_number - window)
        seq = [str(start + j) for j in range(window)]
        data.append(" ".join(seq))
    return "\n".join(data)


a = make_counting_data()
with open('count.txt', 'w')as f:
    f.write(a)
# print(a)
