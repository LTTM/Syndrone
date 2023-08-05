from sys import byteorder


header = """ply
format binary_little_endian 1.0
element vertex %d
property float32 x
property float32 y
property float32 z
property float32 CosAngle
property uint16 ObjIdx
property uint8 ObjTag
end_header
"""


def carla_lidarbuffer_to_ply(filename, npoints, buffer, flip_xy='none'):
    with open(filename, 'w') as f:
        f.write(header%(npoints))
    
    biter = iter(buffer)
    out_bf = bytearray(19*npoints)
    for i in range(npoints):
        x = [next(biter), next(biter), next(biter), next(biter)]
        y = [next(biter), next(biter), next(biter), next(biter)]
        z = [next(biter), next(biter), next(biter), next(biter)]
        a = [next(biter), next(biter), next(biter), next(biter)]
        
        if byteorder == 'little':
            oid = [next(biter), next(biter)]
            _, _ = next(biter), next(biter)
            
            lab = [next(biter)]
            _, _, _ = next(biter), next(biter), next(biter)
            
        else:
            _, _ = next(biter), next(biter)
            oid = [next(biter), next(biter)]
            
            _, _, _ = next(biter), next(biter), next(biter)
            lab = [next(biter)]
            
        if byteorder == 'big':
            x, y, z, a, oid = reversed(x), reversed(y), reversed(z), reversed(a), reversed(oid)
        
        if flip_xy == 'y':
            y[-1] ^= 128 # flip first bit to swap sign
        elif flip_xy == 'x':
            x[-1] ^= 128 # flip first bit to swap sign
        elif flip_xy == 'both':
            x[-1] ^= 128 # flip first bit to swap sign
            y[-1] ^= 128 # flip first bit to swap sign
        
        di = 19*i
        out_bf[di:di+4] = x
        out_bf[di+4:di+8] = y
        out_bf[di+8:di+12] = z
        out_bf[di+12:di+16] = a
        out_bf[di+16:di+18] = oid
        out_bf[di+18:di+19] = lab
        
    with open(filename, 'ab') as f:
        f.write(out_bf)
        