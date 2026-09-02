import logging
import time

from reachy import Reachy, parts
from . import FlyerBackground

import numpy as np
import zzlog

logger = logging.getLogger('reachy.flyers')


def run_distribution_loop(flyer_background):

    tracking_threshold = 20*20 
    action_threshold = 150*150 # 150*150

    track_count = 0
    give_count = 0
    action_count = 0

    flyer_given = 0

    was_in_idle_mode = False
    
    nobody_here = True

    flyer_background.detection.start()
    flyer_background.activate_tracking_mode()
    time.sleep(0.02)
    
    while True:
        
        if flyer_background.detection._somebody_detected :
            
            logger.info(
                'Someone has been detected',
                extra={
                    'x': flyer_background.detection._face_target[0],
                    'y': flyer_background.detection._face_target[1],
                },
            )

            flyer_background.idleForever.stop()
            if was_in_idle_mode:
                flyer_background.reinitialize_target()
                #flyer_background.detect_new_person()
                flyer_background.hand_empty = flyer_background.idle.hand_empty
                was_in_idle_mode = False
            flyer_background.get_target_info()
            
            #if someone is close enough to begin an interaction 
            if flyer_background.target_size > tracking_threshold:
                track_count = 0
                
                #if someone is close, but not too close 
                if flyer_background.target_size < action_threshold:
                    
                    logger.info('Person is not close enough, tracking mode activated',
                        extra={
                            'target-size': flyer_background.target_size,
                            'threshold': action_threshold,
                        })
                    
                    flyer_background.activate_tracking_mode()
                    flyer_background.track()
                    
                    give_count = 0
                    action_count = 0
                    
                #if someone is close enough to start giving the flyer 
                else:

                    if flyer_background.person_comes_for_flyer():
                        
                        stack_im = []
                         
                        #is the person has not been seen before
                        if flyer_background.is_new_person():       
                            stack_im.append(flyer_background.reachy.head.get_image())
                            
                            #give count used to prevent Reachy from giving flyers too frequently 
                            if give_count == 0:
                                
                                flyer_background.deactivate_tracking_mode()
                                
                                if flyer_background.hand_empty:
                                    logger.info('Reachy is ready to give a flyer.')
                                    flyer_background.take_flyer()
                                
                                stack_im.append(flyer_background.reachy.head.get_image())
                                #track = 0 

                                #flyer_background.activate_tracking_mode()

                                # after pulling the flyer, track a bit in case the person has moved
                                #while track < 5:
                                 #   if flyer_background.detection._somebody_detected :
                                  #      flyer_background.get_target_info()
                                   #     flyer_background.track()
                                    #    track += 1
                                     #   time.sleep(0.02)
                                      #  nobody_here = False
                                  #  else:
                                   #     flyer_background.look_at_previous_target()
                                    #    track +=1
                                     #   nobody_here = True
                                
                                #flyer_background.deactivate_tracking_mode()
                                
                                stack_im.append(flyer_background.reachy.head.get_image())
                                
                                #if nobody_here:
                                #    logger.info('The person left, Reachy is sad')
                                #    flyer_background.has_been_ignored()
                                #    flyer_background.reinitialize_target()
                                #    give_count = 5
                                #else:
                                stack_im.append(flyer_background.reachy.head.get_image())

                                flyer_background.give_flyer()

                                stack_im.append(flyer_background.reachy.head.get_image())

                                flyer_given += 1
                                logger.info('Reachy has given the flyer',
                                    extra={
                                        'flyer_number': flyer_given,
                                    }
                                )

                                give_count = 7

                                ind = np.random.randint(1000)

                                flyer_background.emb.add_someone(name='person' + str(ind), ssd_engine=flyer_background.ssd_engine, stack_im=stack_im)

                                logger.info('Added someone',
                                            extra={
                                                'person_name':'person' + str(ind),
                                            }
                                )
                                time.sleep(0.1)

                            give_count -=1
                            time.sleep(0.1)
                            flyer_background.activate_tracking_mode()
                    
                        else:
                            if action_count == 0:
                                flyer_background.deactivate_tracking_mode() #WARNING ADDED
                                flyer_background.no_flyer()
                                action_count = 20
                            action_count -= 1
                            flyer_background.activate_tracking_mode()
                            
                    #track at the end in all cases
                    flyer_background.get_target_info()
                    flyer_background.track()
                        
        else:
            if track_count< 40:
                track_count+=1
            else :
                logger.info('No one detected, Reachy plays random behavior.', extra = {'behavior-played':flyer_background.idleForever.idle_behavior.behavior_played})
                track_count = 0
                flyer_background.deactivate_tracking_mode()
                time.sleep(0.2)         
                flyer_background.idleForever.start()
                was_in_idle_mode = True
            
        
        time.sleep(0.02)


if __name__ == '__main__':

    import argparse

    from datetime import datetime
    from glob import glob

    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file')
    args = parser.parse_args()

    if args.log_file is not None:
        n = len(glob(f'{args.log_file}*.log')) + 1

        now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')
        args.log_file += f'-{n}-{now}.log'

    _ = zzlog.setup(
        logger_root='',
        filename=args.log_file,
    )

    logger.info(
        'Initializing flyer distribution.'
    )

    with FlyerBackground() as flyer_background:
        flyer_background.setup()
        run_distribution_loop(flyer_background)
