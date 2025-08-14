
import numpy as np
right_segment = [['left', 0.8781843457288185], ['left', 0.8829682037906356], ['left', 0.9065054345174818], ['left', 0.8745847435554001], ['left', 0.7923184727572692], ['left', 0.7717240918043768], ['left', 0.8219167557048326], ['left', 0.6361567906866714], ['left', 0.5529131893728024], ['right', 0.6578265730188513], ['right', 0.526713653867863], ['left', 0.6248678260974716], ['left', 0.6810235070506439], ['left', 0.5724417277551417], ['right', 0.5983375774575731], ['right', 0.5356154965516676], ['right', 0.6151643958946829], ['right', 0.6155763006965193], ['right', 0.5925823943234925], ['right', 0.5898184901168634], ['left', 0.4315341165433717], ['left', 0.5519246067727335], ['left', 0.6227433734026969], ['left', 0.7741812435808538]]
left_segment = [['left', 0.9728426844128295], ['left', 0.9763188292205821], ['left', 0.9780148221437639], ['left', 0.9729500504638982], ['left', 0.976199618476432], ['left', 0.972043681548233], ['left', 0.9744418820620825], ['left', 0.9744418275649904], ['left', 0.9720502379179118], ['left', 0.9732949789104599], ['left', 0.9749485945889913], ['left', 0.9749485445241254], ['left', 0.9723091039352483], ['left', 0.9720356647187808], ['left', 0.971828726612853], ['left', 0.9708814274015936], ['left', 0.9718286202157568], ['left', 0.9675827362628917], ['left', 0.9603364728437093], ['left', 0.9736765643042726], ['left', 0.9711624206503491], ['left', 0.9711623790585745], ['left', 0.9688038276563924], ['left', 0.9664055289197464], ['left', 0.9466873801531933]]
flagged_index = ['none', 0.6524236908375741]

# approach one: soft change-point score: 

dir = 'left_to_none'
# np.log is natural log
# np.log10 is log_10
def logit(confidence):
    return np.log(confidence / (1-confidence))

# step 1, signed series
def make_signed_series(stream, direction):
    series = []
    if(direction == 'left_to_none'):
        for arr in stream: 
            if(arr[0] == 'left'):
                series.append(logit(arr[1]))
            elif(arr[0] == 'none'):
                series.append(-1 * logit(arr[1]))
            else: 
                series.append(0)
        return series

# step 2, proximity weighted means
# assuming length of 50, lseg is 25 and rseg is 24
def calculate_directional_term(lseg, rseg, dir):
    if(direction == 'left_to_none'):
        numerator = 0.0
        denominator = 0.0
        # for i in range(len(lseg)):



def calculate_weight(ind_distance, tau = 6.0): 
    return np.exp(-1 * abs(ind_distance)/tau)

print(make_signed_series(left_segment,dir))
print(make_signed_series(right_segment,dir))