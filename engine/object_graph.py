class ObjectGraph:

    def __init__(self, Name: str, Box, 
                 Category : str = "object", 
                 Location : tuple = (0, 0), 
                 Size : tuple = (0, 0)) -> None:
        
        self.Attribute = dict(
            Name = Name,
            Box = Box,
            Category = Category,
            Location = Location,
            Size = Size
        )

    def add(self, key: str, value: any) -> None:

        self.Attribute[key] = value

class ObjectGraphGroup:

    def __init__(self, img=None, groupA=None, groupB=None):
        if img is not None:
            self.Graphs = []
            self.Relations = dict()
            self.Img = img
        elif groupA is not None and groupB is not None:
            self.Graphs = groupA.Graphs + groupB.Graphs
            self.Relations = groupA.Relations | groupB.Relations
            self.Img = groupA.Img
        else:
            raise ValueError("You must give a image or give two groups.")

    def add_graph(self, graph: ObjectGraph):
        
        self.Graphs.append(graph)

    def add_relation(self, objA, objB, relation):
        self.Relations[(objA, objB)] = relation
