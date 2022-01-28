from sklearn import tree
import pydotplus
from IPython.display import Image, display


def show(x, y, feature_names=None, class_names=None):
    model = tree.DecisionTreeClassifier().fit(x, y)
    dot = tree.export_graphviz(model, out_file=None, feature_names=feature_names,
                               class_names=class_names, filled=True, rounded=True,
                               special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot)
    image = Image(graph.create_png())
    display(image)
