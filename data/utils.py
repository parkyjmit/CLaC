# import numpy as np



# ### The adsorbate is attached to the crystal.
# def shortest_length_to_metal(atoms, idx):
#     '''
#     find nearest metal atom from idx th atom
#     '''

#     # 3D 좌표를 나타내는 배열 (예시)
#     atom_coordinates = np.array([
#         [1.0, 2.0, 3.0],  # 첫 번째 원자의 좌표
#         [4.0, 5.0, 6.0],  # 두 번째 원자의 좌표
#         # ... 추가적인 원자들의 좌표
#     ])

#     # 거리 매트릭스 초기화
#     distance_matrix = np.linalg.norm(positions.unsqueeze(0) - positions.unsqueeze(1), axis=-1)

#     # i번째 원자에 대한 가장 가까운 다른 원자의 인덱스 찾기

#     closest_atom_index = np.argmin([d for i, d in enumerate(distance_matrix[idx]) if i != idx])