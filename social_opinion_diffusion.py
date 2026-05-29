import random
import networkx as nx
import matplotlib.pyplot as plt

# 创建一个社交网络图，节点代表用户，边代表好友关系
G = nx.erdos_renyi_graph(n=20, p=0.15, seed=42)

# 初始负面舆论源，从一个用户开始传播
seed_node = 0
infected = {seed_node}
frontier = {seed_node}

# 逐层传播：每一轮，感染当前感染者的邻居
while frontier:
    next_frontier = set()
    for node in frontier:
        for neighbor in G.neighbors(node):
            if neighbor not in infected:
                next_frontier.add(neighbor)
    if not next_frontier:
        break
    infected |= next_frontier
    frontier = next_frontier

# 生成节点颜色：红色表示感染，蓝色表示中立
colors = ["red" if node in infected else "blue" for node in G.nodes()]
pos = nx.spring_layout(G, seed=1)

# 绘制网络图并显示节点编号
plt.figure(figsize=(8, 6))
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500, alpha=0.9)
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_color="white")
plt.title("社交网络舆论情绪扩散模拟")
plt.axis("off")
plt.tight_layout()
plt.show()

# 终端输出统计信息
infected_count = len(infected)
total_count = G.number_of_nodes()
spread_rate = infected_count / total_count
print(f"感染人数: {infected_count} / {total_count}")
print(f"舆论扩散率: {spread_rate:.2%}")

# VS Code 终端运行命令示例
# python social_opinion_diffusion.py
