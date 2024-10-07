import numpy as np

class node_state(object):
    def __init__(self, matrix, lowerBound, row, process):
        self.matrix = matrix            # 当前状态下的矩阵
        self.lowerBound = lowerBound    # 下界
        self.row = row                  # 当前工人（行）
        self.process = process          # 已分配工程所需的时间累积总和


# 堆结构调整：向上调整（每次插入节点后需要进行的调整）
def fix_up(queue):
    temp_idx = len(queue) - 1           # 选中堆末尾的节点
    parent_idx = (temp_idx - 1) // 2    # 获取其父节点
    heap = False                        # 标志位：True-堆结构调整完成；False-未完成
    while not heap and parent_idx >= 0: # 循环条件：堆结构调整未完成 且 父节点依旧存在（未遍历完队列）
        # 判断是否满足堆的性质
        if queue[temp_idx].lowerBound < queue[parent_idx].lowerBound:
            queue[temp_idx], queue[parent_idx] = queue[parent_idx], queue[temp_idx]
        else:
            heap = True
        # 更新索引值，继续向上调整
        temp_idx = parent_idx
        parent_idx = (temp_idx - 1) // 2

# 堆结构调整：向下调整（每次交换堆头与堆尾元素并出队后的调整）
def fix_down(queue):
    if not queue:
        return
    temp_idx = 0
    size = len(queue)
    temp_ver = queue[temp_idx]                  # 暂存当前节点
    heap = False                                # 标志位：True-堆结构调整完成；False-未完成
    while not heap and 2 * temp_idx + 1 < size: # 循环条件：堆结构调整未完成 且 左孩子存在
        j = 2 * temp_idx + 1                    # 左孩子的索引
        # 左孩子存在
        if j < size - 1:
            # 比较两个孩子节点的权重（下界），记录权重更小的节点
            if queue[j].lowerBound > queue[j + 1].lowerBound:
                j = j + 1
        # 判断是否满足堆的性质
        if queue[j].lowerBound >= temp_ver.lowerBound:
            heap = True
        else:
            queue[temp_idx] = queue[j]
            temp_idx = j                        # 更新索引值，继续向下调整
    queue[temp_idx] = temp_ver


def main(matrix):
    row, col = matrix.shape                                                 # 获取矩阵的行、列大小
    heapqueue = []                                                          # 声明优先队列
    cur_node_state = node_state(matrix, 0, 0, 0)    # 初始状态
    heapqueue.append(cur_node_state)                                        # 根节点入队

    while cur_node_state.row < row - 1:
        cur_node_state = heapqueue.pop()        # 出队
        fix_down(heapqueue)                     # 向下调整堆结构
        cur_matrix = cur_node_state.matrix      # 获取当前状态下的初始矩阵
        cur_row = cur_node_state.row            # 获取当前状态所在的行

        for cur_col in range(col):
            if cur_matrix[cur_row][cur_col] != np.inf:
                temp_matrix = cur_matrix.copy()     # 复制当前状态下的初始矩阵
                temp_matrix[cur_row, :] = np.inf    # 用 ∞ 划掉已被分配的列
                temp_matrix[:, cur_col] = np.inf    # 用 ∞ 划掉已被分配的行

                # 计算下界
                lower_bound = (
                    sum([min(worktime) for worktime in temp_matrix[cur_row + 1 :]])  # 加和当前行以下（不包括当前行）的每行最小工作时间
                    + cur_node_state.process                                         # 加和当前行之前（不包括当前行）的累积最小工作时间
                    + cur_matrix[cur_row][cur_col]                                   # 加和当前行、当前列的工作时间
                )

                # 更新已处理行的累积值
                process = (
                    cur_node_state.process + cur_matrix[cur_row][cur_col]
                )

                heapqueue.append(node_state(temp_matrix, lower_bound, cur_row + 1, process))    # 当前节点入队
                fix_up(heapqueue)                                                               # 向上调整堆结构

        heapqueue[0], heapqueue[-1] = heapqueue[-1], heapqueue[0]   # 交换堆顶元素（根节点）与最后一个元素 （堆出队的必要操作）
    return cur_node_state.lowerBound


# 测试矩阵
test_matrix = np.array([[12, 8, 10, 9,],
                        [9, 7, 6, 10],
                        [8, 11, 4, 11],
                        [7, 9, 12, 7]], dtype=np.float32)

print("The minimal cost is %d." % main(test_matrix))