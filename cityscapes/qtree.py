import matplotlib.pyplot as plt
from typing import List


class Point:
    def __init__(self, x: float, y: float, tag: int):
        self.x = x
        self.y = y
        self.tag = tag


class Node:
    def __init__(self, x0: float, y0: float, w: float, h: float, points: List[Point]):
        self.x0 = x0
        self.y0 = y0
        self.width = w
        self.height = h
        self.points = points
        self.children = []

    def get_width(self) -> float:
        return self.width

    def get_height(self) -> float:
        return self.height

    def get_points(self) -> List[Point]:
        return self.points


class QTree:
    def __init__(self, k: int, n: int):
        self.threshold = k
        self.points = []
        self.root = Node(0, 0, n, n, self.points)

    def add_point(self, x: float, y: float, tag: int):
        self.points.append(Point(x, y, tag))

    def get_points(self) -> List[Point]:
        return self.points

    def subdivide(self):
        recursive_subdivide(self.root, self.threshold)

    def graph(self):
        fig = plt.figure(figsize=(12, 8))
        plt.title("Quadtree")
        c = find_children(self.root)
        print("Number of segments: %d" % len(c))
        areas = set()
        for el in c:
            areas.add(el.width * el.height)
        print("Minimum segment area: %.3f units" % min(areas))
        for n in c:
            plt.gcf().gca().add_patch(
                plt.Rectangle((n.x0, n.y0), n.width, n.height, fill=False)
            )
        x = [point.x for point in self.points]
        y = [point.y for point in self.points]
        plt.plot(x, y, "ro")
        plt.show()
        return

    def plot(self, ax):
        ax.set_title("Quadtree Visualisation")
        ax.axis("equal")
        c = find_children(self.root)
        print("Number of segments: %d" % len(c))
        areas = set()
        for el in c:
            areas.add(el.width * el.height)
        print("Minimum segment area: %.3f units" % min(areas))
        for n in c:
            ax.add_patch(plt.Rectangle((n.x0, n.y0), n.width, n.height, fill=False))
        x = [point.x for point in self.points]
        y = [point.y for point in self.points]
        ax.plot(x, y, "ro")
        return ax

    def search(self, x: float, y: float, d: float) -> List[Point]:
        return recursive_search(self.root, Point(x, y), d)


def recursive_subdivide(node: Node, k: int):
    if len(node.points) <= k:
        return

    w_ = float(node.width / 2)
    h_ = float(node.height / 2)

    p = contains(node.x0, node.y0, w_, h_, node.points)
    x1 = Node(node.x0, node.y0, w_, h_, p)
    recursive_subdivide(x1, k)

    p = contains(node.x0, node.y0 + h_, w_, h_, node.points)
    x2 = Node(node.x0, node.y0 + h_, w_, h_, p)
    recursive_subdivide(x2, k)

    p = contains(node.x0 + w_, node.y0, w_, h_, node.points)
    x3 = Node(node.x0 + w_, node.y0, w_, h_, p)
    recursive_subdivide(x3, k)

    p = contains(node.x0 + w_, node.y0 + h_, w_, h_, node.points)
    x4 = Node(node.x0 + w_, node.y0 + h_, w_, h_, p)
    recursive_subdivide(x4, k)

    node.children = [x1, x2, x3, x4]


def contains(
    x: float, y: float, w: float, h: float, points: List[Point]
) -> List[Point]:
    pts = []
    for point in points:
        if point.x >= x and point.x <= x + w and point.y >= y and point.y <= y + h:
            pts.append(point)
    return pts


def find_children(node: Node) -> List[Node]:
    if not node.children:
        return [node]
    else:
        children = []
        for child in node.children:
            children += find_children(child)
    return children


def recursive_search(node: Node, point: Point, dist: float) -> List[Point]:
    if not node.children:
        return [
            p
            for p in node.points
            if (p.x - point.x) ** 2 + (p.y - point.y) ** 2 <= dist**2
        ]

    results = []
    for child in node.children:
        if (
            point.x >= child.x0 - dist
            and point.x <= child.x0 + child.width + dist
            and point.y >= child.y0 - dist
            and point.y <= child.y0 + child.height + dist
        ):
            results.extend(recursive_search(child, point, dist))

    return results
