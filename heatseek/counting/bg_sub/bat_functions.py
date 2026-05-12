import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import linear_sum_assignment
from CountLine import CountLine
import koger_tracking as ktf 

def get_blob_info(binary_image, background=None, size_threshold=0):
    
    '''
    Get contours from binary image. Then find center and average radius of each contour
    
    binary_image: 2D image
    background: 2D array used to see locally how dark the background is
    size_threshold: radius above which blob is considered real
    '''
    
    contours, hierarchy = cv2.findContours(binary_image.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    # Size of bounding rectangles
    sizes = []
    areas = []
    # angle of bounding rectangle
    angles = []
    rects = []
    good_contours = []
    contours = [np.squeeze(contour) for contour in contours]
    
    for contour_ind, contour in enumerate(contours):
        
        
        
        if len(contour.shape) >  1:
            
            rect = cv2.minAreaRect(contour)
            
            if background is not None:
                darkness = background[int(rect[0][1]), int(rect[0][0])]
                if darkness < 30:
                    dark_size_threshold = size_threshold + 22
                elif darkness < 50:
                    dark_size_threshold = size_threshold + 15
                elif darkness < 80:
                    dark_size_threshold = size_threshold + 10
                elif darkness < 100:
                    dark_size_threshold = size_threshold + 5
    #             elif darkness < 130:
    #                 dark_size_threshold = size_threshold + 3
                else:
                    dark_size_threshold = size_threshold
            else:
                dark_size_threshold = 0 # just used in if statement

            area = rect[1][0] * rect[1][1]
            
            if (area >= dark_size_threshold) or background is None:
                centers.append(rect[0])
                sizes.append(rect[1])
                angles.append(rect[2])
                good_contours.append(contour)
                areas.append(area)
                rects.append(rect)
    if centers:
        centers = np.stack(centers, 0)
        sizes = np.stack(sizes, 0)
    else:
        centers = np.zeros((0,2))
            
    return (centers, np.array(areas), good_contours, angles, sizes, rects)

    
def add_all_points_as_new_tracks(raw_track_list, positions, contours, 
                                 sizes, current_frame_ind, noise):
    """ When there are no active tracks, add all new points to new tracks.
    
    Args:
        raw_track_list (list): list of tracks
        positions (numpy array): p x 2
        contours (list): p contours
        current_frame_ind (int): current frame index
        noise: how much noise to add to tracks initially
    """
    
    for ind, (position, contour, size) in enumerate(zip(positions, contours, sizes)):
        raw_track_list.append(
            ktf.create_new_track(first_frame=current_frame_ind, 
                                 first_position=position, pos_index=ind, 
                                 noise=noise, contour=contour, size=size 
            )
        )
        
    return raw_track_list
    
    

def find_tracks(first_frame_ind, positions, 
                contours_files=None, contours_list=None,
                sizes_list=None, max_frame=None, verbose=True, 
                tracks_file=None):
    """ Take in positions of all individuals in frames and find tracks.
    
    Args: 
        first_frame_ind (int): index of first frame of these tracks
        positions (list): n x 2 for each frame
        contours_files (list): list of files for contour info from each frame
        contours_list: already loaded list of contours, only used if contours_file
            is None
        sizes_list (list): sizes info from each frame
    
    return list of all tracks found
    """
    
    raw_track_list = []

    max_distance_threshold = 30
    max_distance_threshold_noise = 30
    min_distance_threshold = 0
    max_unseen_time = 2
    min_new_track_distance = 3
    min_distance_big = 30

#     #Create initial tracks based on the objects in the first frame
#     raw_track_list = add_all_points_as_new_tracks(
#         raw_track_list, positions[0], contours_list[0], sizes_list0, noise=0
#     )

    #try to connect points to the next frame
    if max_frame is None:
        max_frame = len(positions)
        
    contours_file_ind = 0
    previous_contours_seen = 0
    if contours_files:
        contours_list = np.load(contours_files[contours_file_ind], allow_pickle=True)
        while first_frame_ind >= previous_contours_seen + len(contours_list):
            contours_file_ind += 1
            previous_contours_seen += len(contours_list)
            contours_list = np.load(contours_files[contours_file_ind], allow_pickle=True)
        print(f'using {contours_files[contours_file_ind]}')   
    elif not contours_list:
        print("Needs contour_files or contour_list")
        return
        
    
    contours_ind = first_frame_ind - previous_contours_seen - 1
    
    
    for frame_ind in range(first_frame_ind, max_frame):
        contours_ind += 1
        
        if contours_files:
            if contours_ind >= len(contours_list):
                # load next file
                try:
                    contours_file_ind += 1
                    contours_list = np.load(contours_files[contours_file_ind], allow_pickle=True)
                    contours_ind = 0
                except:
                    if tracks_file:
                        tracks_file_error = os.path.splitext(tracks_file)[0] + f'-error-{frame_ind}.npy'
                        print(tracks_file_error)
                        np.save(tracks_file_error, np.array(raw_track_list, dtype=object))
        #get tracks that are still active (have been seen within the specified time)
        active_list = ktf.calculate_active_list(raw_track_list, max_unseen_time, frame_ind)
        
        if verbose:
            if frame_ind % 10000 == 0:
                print('frame {} processed.'.format(frame_ind))
                if tracks_file:
                    np.save(tracks_file, np.array(raw_track_list, dtype=object))
        if len(active_list) == 0:
            #No existing tracks to connect to
            #Every point in next frame must start a new track
            raw_track_list = add_all_points_as_new_tracks(
                raw_track_list, positions[frame_ind], contours_list[contours_ind], 
                sizes_list[frame_ind], frame_ind, noise=1
            )
            continue

        # Make sure there are new points to add
        new_positions = None
        row_ind = None
        col_ind = None
        new_sizes = None
        new_position_indexes = None
        distance = None
        contours = None
        if len(positions[frame_ind]) != 0:
            
            #positions from the next step
            new_positions = positions[frame_ind]
            contours = [np.copy(contour) for contour in contours_list[contours_ind]]
            new_sizes = sizes_list[frame_ind]
            
            raw_track_list = ktf.calculate_max_distance(
                raw_track_list, active_list, max_distance_threshold, 
                max_distance_threshold_noise, min_distance_threshold,
                use_size=True, min_distance_big=min_distance_big
            )

            distance = ktf.calculate_distances(
                new_positions, raw_track_list, active_list
            )
            
            max_distance = ktf.create_max_distance_array(
                distance, raw_track_list, active_list
            )
            
            assert distance.shape[1] == len(new_positions)
            assert distance.shape[1] == len(contours)
            assert distance.shape[1] == len(new_sizes)
                
            # Some new points could be too far away from every existing track
            raw_track_list, distance, new_positions, new_position_indexes, new_sizes, contours = ktf.process_points_without_tracks(
                distance, max_distance, raw_track_list, new_positions, contours, 
                frame_ind, new_sizes
            )
            
                
            if distance.shape[1] > 0:
                # There are new points can be assigned to existing tracks
                #connect the dots from one frame to the next
                
                row_ind, col_ind = linear_sum_assignment(np.log(distance + 1))
                
#                 for active_ind, track_ind in enumerate(active_list):
#                     if active_ind in row_ind:
#                         row_count = np.where(row_ind == active_ind)[0]
#                         raw_track_list[track_ind]['debug'].append(
#                             '{} dist {},  best {}'.format(
#                                 frame_ind,
#                                 distance[row_ind[row_count],
#                                          col_ind[row_count]],
#                                 np.min(distance[row_ind[row_count],
#                                          :])
#                             )
#                         )
#                         best_col = np.argmin(distance[row_ind[row_count],
#                                          :])
#                         row_count = np.where(col_ind == best_col)[0]
#                         raw_track_list[track_ind]['debug'].append(
#                             '{} row_ind {} col {} dist {} track {}'.format(
#                             frame_ind, row_ind[row_count],
#                             col_ind[row_count],
#                             distance[row_ind[row_count],
#                                          col_ind[row_count]],
#                             active_list[row_ind[row_count][0]])
#                         )
                        
                
                # In casese where there are fewer new points than existing tracks
                # some tracks won't get new point. Just assign them to 
                # the closest point
                row_ind, col_ind = ktf.filter_tracks_without_new_points(
                    raw_track_list, distance, row_ind, col_ind, active_list, frame_ind
                )
                # Check if tracks with big bats got assigned to small points which are
                # probably noise
                row_ind, col_ind = ktf.fix_tracks_with_small_points(
                    raw_track_list, distance, row_ind, col_ind, active_list, new_sizes, frame_ind)
                # see if points got assigned to tracks that are farther 
                # than max_threshold_distance
                # This happens when the closer track gets assigned 
                # to a differnt point
                row_ind, col_ind = ktf.filter_bad_assigns(raw_track_list, active_list, distance, max_distance,
                                                      row_ind, col_ind
                                                     )


        raw_track_list = ktf.update_tracks(raw_track_list, active_list, frame_ind, 
                                           row_ind, col_ind, new_positions, 
                                           new_position_indexes, new_sizes, contours, 
                                           distance, min_new_track_distance)
        raw_track_list = ktf.remove_noisy_tracks(raw_track_list)
    raw_track_list = ktf.finalize_tracks(raw_track_list) 
    if tracks_file:
        np.save(tracks_file, np.array(raw_track_list, dtype=object))
        print('{} final save.'.format(os.path.basename(os.path.dirname(tracks_file)))) 
    return raw_track_list

def threshold_short_tracks(raw_track_list, min_length_threshold=2):
    """Only return tracks that are longer than min_length_threshold."""
    
    track_list = []
    for track_num, track in enumerate(raw_track_list):
        if isinstance(track['track'], list):
            track['track'] = np.array(track['track'])
        track_length = track['track'].shape[0]
        if track_length >= min_length_threshold:
            track_list.append(track)
    return track_list

def get_rects(track):
    """ Fit rotated bounding rectangles to each contour in track.
    
    track: track dict with 'contour' key linked to list of cv2 contours
    """
    rects = []
    for contour in track['contour']:
        if len(contour.shape) >  1:
            rect = cv2.minAreaRect(contour)
            rects.append(rect[1])
        else:
            rects.append((np.nan, np.nan))
        
    return np.array(rects)

def get_wingspan(track):
    """ Estimate wingspan in pixels from average of peak sizes of longest
    rectangle edges.
    """
    
    if not 'rects' in track.keys():
        track['rects'] = get_rects(track)
                    
    max_edge = np.nanmax(track['rects'], 1)
    max_edge = max_edge[~np.isnan(max_edge)]
    peaks = signal.find_peaks(max_edge)[0]
    if len(peaks) != 0:
        mean_wing = np.nanmean(max_edge[peaks])
    else:
        mean_wing = np.nanmean(max_edge)
    
    return mean_wing

def measure_crossing_bats(track_list, frame_height=None, frame_width=None,
                          count_across=False, count_out=True, num_frames=None, 
                          with_rects=True, ):
    
    """ Find and quantify all tracks that cross middle line.
    
    track_list: list of track dicts
    frame_height: height of frame in pixels
    frame_width: width of frame in pixels
    count_across: count horizontal tracks
    count_out: count vertical tracks
    num_frames: number of frames in observation
    with_rects: if True calculate rects if not already
        in track and estimate wingspan and body size
    
    """
    if count_across:
        assert frame_width, "If vertical must specify frame width."
        across_line = CountLine(int(frame_width/2), line_dim=0, total_frames=num_frames)
    if count_out:
        assert frame_height, "If horizontal must specify frame height."
        out_line = CountLine(int(frame_height/2), line_dim=1, total_frames=num_frames)

    crossing_track_list = []

    for track_ind, track in enumerate(track_list):
        out_result = None
        across_result = None
        if count_out:
            out_result, out_frame_num = out_line.is_crossing(track, track_ind)
        if count_across:
            across_result, across_frame_num = across_line.is_crossing(track, track_ind)
        if out_result or across_result:
            crossing_track_list.append(track)
            # result is 1 if forward crossing -1 is backward crossing
            if count_out:
                if out_frame_num:
                    crossing_track_list[-1]['crossed'] = out_frame_num * out_result
                else:
                    crossing_track_list[-1]['crossed'] = 0
            if count_across:
                if across_frame_num:
                    crossing_track_list[-1]['across_crossed'] = across_frame_num * across_result
                else:
                    crossing_track_list[-1]['across_crossed'] = 0
            track[id] = track_ind
            if with_rects:
                if not 'rects' in track.keys():
                    track['rects'] = get_rects(track)

                crossing_track_list[-1]['mean_wing'] = get_wingspan(track)

            
    return crossing_track_list

