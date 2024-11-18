from helmoltz_2d_BEM.geometry import Disc
import numpy as np
import matplotlib.pyplot as plt




class BSPNode:
    def __init__(self, bbox: tuple|list|np.ndarray, depth: int = 0) -> None:
        """Initialization of a BSP node.

        Parameters
        ----------
        bbox : tuple | list | np.ndarray
            Bounding box of the node.
        depth : int, optional
            Depth of the node in a BSP tree, by default 0
        """
        self.depth = depth
        self.bbox = np.asarray(bbox)
        self.extents = np.diff(self.bbox).flatten()
        self.largest_dimension_axis = np.argmax(self.extents)
        self.points = []
        self.left_children = None
        self.right_children = None
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if the current node contains the point.

        Parameters
        ----------
        point : np.ndarray
            Considered point.

        Returns
        -------
        bool
            True if the current node contains the considered point.
        """
        # if 0 < point[0] - self.bbox[0, 0] <= self.extents[0] and 0 < point[1] - self.bbox[1, 0] <= self.extents[1]:
        #     return True
        # else: 
        #     return False
        return np.all((self.bbox[:, 0] <= point) & (point <= self.bbox[:, 1]))
        
    
    def split(self) -> None:
        if self.left_children is not None or self.right_children is not None:
            return
        bbox = self.bbox
        if self.largest_dimension_axis == 0:
            mid_x = bbox[0, 0] + self.extents[0] / 2
            bbox_left = [[bbox[0, 0], mid_x], bbox[1]]
            bbox_right = [[mid_x, bbox[0, 1]], bbox[1]]
        elif self.largest_dimension_axis == 1:
            mid_y = bbox[1, 0] + self.extents[1] / 2
            bbox_left = [bbox[0], [bbox[1, 0], mid_y]]
            bbox_right = [bbox[0], [mid_y, bbox[1, 1]]]
        else:
            raise ValueError("Invalid axis for splitting.")

        self.left_children = BSPNode(bbox=bbox_left, depth=self.depth + 1)
        self.right_children = BSPNode(bbox=bbox_right, depth=self.depth + 1)
        
        
    def display(self, ax: plt.axes=None) -> None:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()
        rect = plt.Rectangle(
            (self.bbox[0, 0], self.bbox[1, 0]),
            self.extents[0],
            self.extents[1],
            edgecolor="blue",
            facecolor="none",
            linewidth=0.5
        )
        ax.add_patch(rect)

        # Appel récursif pour les enfants
        if self.left_children:
            self.left_children.display(ax)
        if self.right_children:
            self.right_children.display(ax)

        ax.set_aspect('equal', 'box')
        ax.set_xlim(self.bbox[0])
        ax.set_ylim(self.bbox[1])
    

class BSPTree:
    def __init__(self, points) -> None:
        pass
    
    def _get_children_containing(self, node: BSPNode, point: np.ndarray) -> BSPNode:
        """Get the children of the considered node that contains the considered point.
        
        NOTE This methods assumes that the considered node contains the considered point.

        Parameters
        ----------
        node : BSPNode
            _description_
        point : np.ndarray
            _description_

        Returns
        -------
        BSPNode
            _description_
        """
        if not node.contains(point):
            raise ValueError("Current node does not contains the point so cannot find the children containing the point.")
        
        if node.left_children is None: # no children
            return None
        contains = [node.left_children.contains(point), node.right_children.contains(point)]
        if np.all(contains): # meaning that the point is on the boundary between the childrens so we take the right one (the left one doesn't include the last value in the largest dimension axis)
            return node.left_children
        return np.argmax(contains)

    
    def _insert(self, point):
        pass
    
    def build(self):
        pass
    
    def display(self, ax: plt.axes=None) -> None:
        pass
    
if __name__ == '__main__':
    
    a = np.full(2, fill_value=True)
    a = [False, False]
    print(np.argmax(a))
    # a[0] = False
    print(np.all(a))