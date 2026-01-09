## -- Euler Angles computations -- ## 
import numpy as np

def dcm2eul(R=None, seq=None):
    '''
    Computes the Euler angles from a Rotation Matrix. Formulae are taken from https://ntrs.nasa.gov/api/citations/19770024290/downloads/19770024290.pdf
    The sequence (`seq`) must be one among the 12 allowed ones, being either proper Euler Angles or Tait-Bryan Angles.
    Angles are returned in **radians** [rad]. To convert them to degrees [deg], either use `np.degrees(eul)`, or multiply by 180/π.
    Input:
        - R (np.ndarray(3,3)): Rotation Matrix representing the orientation of the body relative to the frame.
        - seq (str): Sequence of rotation axes. Must be all capital.

    Returns:
        - eul (np.array(3,) - measure unit [rad]): Euler Angles computed from `R` using `seq`.
    '''
    # These are the 12 allowed sequences of rotations
    allowed_sequences = ('XYX', 'YXY', 'ZXZ', 'XZX', 'YZY', 'ZYZ', # Proper Euler Angles
                         'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX') # Tait-Bryan Angles (RPY)
    
    if seq in allowed_sequences:
        r11, r12, r13 = R[0]
        r21, r22, r23 = R[1]
        r31, r32, r33 = R[2]
        eul = np.zeros(3)

        if seq == 'XYZ':
            eul[0] = np.arctan2(-r23, r33)
            eul[1] = np.arctan2(r13, np.sqrt(1-r13**2))
            eul[2] = np.arctan2(-r12, r11)
        
        if seq == 'XZY':
            eul[0] = np.arctan2(r32, r22)
            eul[1] = np.arctan2(-r12, np.sqrt(1-r12**2))
            eul[2] = np.arctan2(r13, r11)

        if seq == 'XYX':
            eul[0] = np.arctan2(r21, -r31)
            eul[1] = np.arctan2(np.sqrt(1-r11**2), r11)
            eul[2] = np.arctan2(r12, r13)

        if seq == 'XZX':
            eul[0] = np.arctan2(r31, r21)
            eul[1] = np.arctan2(np.sqrt(1-r11**2), r11)
            eul[2] = np.arctan2(r13, -r12)

        if seq == 'YXZ':
            eul[0] = np.arctan2(r31, r33)
            eul[1] = np.arctan2(-r23, np.sqrt(1-r23**2))
            eul[2] = np.arctan2(r21, r22)

        if seq == 'YZX':
            eul[0] = np.arctan2(-r31, r11)
            eul[1] = np.arctan2(r21, np.sqrt(1-r21**2))
            eul[2] = np.arctan2(-r23, r22)

        if seq == 'YXY':
            eul[0] = np.arctan2(r12, r32)
            eul[1] = np.arctan2(np.sqrt(1-r22**2), r22)
            eul[2] = np.arctan2(r21, -r23)

        if seq == 'YZY':
            eul[0] = np.arctan2(r32, -r12)
            eul[1] = np.arctan2(np.sqrt(1-r22**2), r22)
            eul[2] = np.arctan2(r23, r21)

        if seq == 'ZXY':
            eul[0] = np.arctan2(-r12, r22)
            eul[1] = np.arctan2(r32, np.sqrt(1-r32**2))
            eul[2] = np.arctan2(-r31, r33)

        if seq == 'ZYX':
            eul[0] = np.arctan2(r21, r11)
            eul[1] = np.arctan2(-r31, np.sqrt(1-r31**2))
            eul[2] = np.arctan2(r32, r33)
   
        if seq == 'ZXZ':
            eul[0] = np.arctan2(r13, -r23)
            eul[1] = np.arctan2(np.sqrt(1-r33**2), r33)
            eul[2] = np.arctan2(r31, r32)

        if seq == 'ZYZ':
            eul[0] = np.arctan2(r23, r13)
            eul[1] = np.arctan2(np.sqrt(1-r33**2), r33)
            eul[2] = np.arctan2(r32, -r31)

    else:
        print(f'Wrong sequence. The argument must be one among the following:\n{allowed_sequences}')
        eul = None
    
    return eul