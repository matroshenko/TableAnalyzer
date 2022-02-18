class Interval(object):
    def __init__(self, start, end):
        assert start < end
        self.start = start
        self.end = end

    def get_length(self):
        return self.end - self.start

    def get_center(self):
        return (self.start + self.end) // 2

    def intersects(self, other):
        return self.start < other.end and other.start < self.end

    def __or__(self, other):
        return Interval(min(self.start, other.start), max(self.end, other.end))

    @staticmethod
    def get_intersection_length(first, second):
        return max(0, min(first.end, second.end) - max(first.start, second.start))

  
def get_intervals_of_ones(mask):
    result = []
    current_inteval_start = None
    is_inside_interval = False
    for i in range(len(mask)):
        if mask[i] == 1:
            if not is_inside_interval:
                current_inteval_start = i
                is_inside_interval = True
        else:
            if is_inside_interval:
                assert current_inteval_start is not None
                result.append(Interval(current_inteval_start, i))
                is_inside_interval = False
    if is_inside_interval:
        assert current_inteval_start is not None
        result.append(Interval(current_inteval_start, len(mask)))
    return result