import numpy as np
from scipy import ndimage
from scipy.ndimage import measurements

from gym_go import govars

##############################################
# All set operations are in-place operations
##############################################

batch_surround_struct = np.array([[[0, 1, 0],
                                   [1, 0, 1],
                                   [0, 1, 0]]])

batch_binary_struct = np.array([[[0, 0, 0],
                                 [0, 0, 0],
                                 [0, 0, 0]],
                                [[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]],
                                [[0, 0, 0],
                                 [0, 0, 0],
                                 [0, 0, 0]]])


def get_group_map(state: np.ndarray):
    group_map = [set(), set()]
    all_pieces = np.sum(state[[govars.BLACK, govars.WHITE]], axis=0)
    for player in [govars.BLACK, govars.WHITE]:
        pieces = state[player]
        labels, num_groups = measurements.label(pieces)
        for group_idx in range(1, num_groups + 1):
            group = govars.Group()

            group_matrix = (labels == group_idx)
            liberty_matrix = ndimage.binary_dilation(group_matrix) * (1 - all_pieces)
            liberties = np.argwhere(liberty_matrix)
            for liberty in liberties:
                group.liberties.add(tuple(liberty))

            locations = np.argwhere(group_matrix)
            for loc in locations:
                loc = tuple(loc)
                group.locations.add(loc)
                group_map[player].add(group)

    return group_map


def get_batch_invalid_moves(states, group_map, player):
    """
    Does not include ko-protection and assumes it will be taken care of elsewhere
    Updates invalid moves in the OPPONENT's perspective
    1.) Opponent cannot move at a location
        i.) If it's occupied
        i.) If it's protected by ko
    2.) Opponent can move at a location
        i.) If it can kill
    3.) Opponent cannot move at a location
        i.) If it's adjacent to one of their groups with only one liberty and
            not adjacent to other groups with more than one liberty and is completely surrounded
        ii.) If it's surrounded by our pieces and all of those corresponding groups
            move more than one liberty
    """

    all_pieces = np.sum(states[:, [govars.BLACK, govars.WHITE]], axis=1)

    # Possible invalids are on single liberties of opponent groups and on multi-liberties of own groups
    invalid_array = get_batch_possible_invalids(states, group_map, player)

    surrounded = ndimage.convolve(all_pieces, batch_surround_struct, mode='constant', cval=1) == 4

    return surrounded * invalid_array + all_pieces


def get_batch_possible_invalids(states, group_maps, player):
    invalid_array = np.zeros((states.shape[0], states.shape[2], states.shape[3]))
    for i in range(len(states)):
        possible_invalids = set()
        definite_valids = set()
        own_groups = group_maps[i][player]
        opp_groups = group_maps[i][1 - player]
        for group in opp_groups:
            if len(group.liberties) == 1:
                possible_invalids.update(group.liberties)
            else:
                # Can connect to other groups with multi liberties
                definite_valids.update(group.liberties)
        for group in own_groups:
            if len(group.liberties) > 1:
                possible_invalids.update(group.liberties)
            else:
                # Can kill
                definite_valids.update(group.liberties)
        possible_invalids.difference_update(definite_valids)

        for loc in possible_invalids:
            invalid_array[i, loc[0], loc[1]] = 1
    return invalid_array


def get_batch_adj_data(state, batch_locs):
    batch_size = len(batch_locs)
    all_pieces = np.sum(state[[govars.BLACK, govars.WHITE]], axis=0)
    surrounded = ndimage.convolve(all_pieces, batch_surround_struct[0], mode='constant', cval=1)

    locs_array = np.zeros((batch_size,) + state.shape[1:])
    locs_array[np.arange(batch_size), batch_locs[:, 0], batch_locs[:, 1]] = 1
    batch_surrounded = surrounded[batch_locs[:, 0], batch_locs[:, 1]] == 4

    dilated = ndimage.binary_dilation(locs_array, batch_binary_struct)
    neighbors = dilated - locs_array
    neighbor_locs = np.argwhere(neighbors)

    batch_tuple_adj_locs = []
    curr_idx = -1
    for adj_locs in neighbor_locs:
        if adj_locs[0] > curr_idx:
            batch_tuple_adj_locs.append([])
            curr_idx += 1
        batch_tuple_adj_locs[-1].append(tuple(adj_locs[1:]))
    if len(batch_tuple_adj_locs) < len(batch_locs):
        batch_tuple_adj_locs.append(None)
    return batch_tuple_adj_locs, batch_surrounded


def get_adjacent_groups(group_map, adjacent_locations, player):
    our_groups, opponent_groups = set(), set()
    for adj_loc in adjacent_locations:
        found = False
        for group in group_map[player]:
            if adj_loc in group.locations:
                our_groups.add(group)
                found = True
                break

        if not found:
            for group in group_map[1 - player]:
                if adj_loc in group.locations:
                    opponent_groups.add(group)
                    break
    return our_groups, opponent_groups


def get_liberties(state: np.ndarray):
    blacks = state[govars.BLACK]
    whites = state[govars.WHITE]
    all_pieces = np.sum(state[[govars.BLACK, govars.WHITE]], axis=0)

    liberty_list = []
    for player_pieces in [blacks, whites]:
        liberties = ndimage.binary_dilation(player_pieces)
        liberties *= (1 - all_pieces).astype(np.bool)
        liberty_list.append(liberties)

    return liberty_list[0], liberty_list[1]


def get_num_liberties(state: np.ndarray):
    '''
    :param state:
    :return: Total black and white liberties
    '''
    black_liberties, white_liberties = get_liberties(state)
    black_liberties = np.count_nonzero(black_liberties)
    white_liberties = np.count_nonzero(white_liberties)

    return black_liberties, white_liberties


def get_board_size(state):
    assert state.shape[1] == state.shape[2]
    return (state.shape[1], state.shape[2])


def get_turn(state):
    """
    Returns who's turn it is (govars.BLACK/govars.WHITE)
    :param state:
    :return:
    """
    return int(state[govars.TURN_CHNL, 0, 0])


def batch_set_turn(states):
    states[:, govars.TURN_CHNL] = 1 - states[:, govars.TURN_CHNL]


def set_turn(state):
    """
    Swaps turn
    :param state:
    :return:
    """
    state[govars.TURN_CHNL] = 1 - state[govars.TURN_CHNL]
