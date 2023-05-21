import opensim as osim
import numpy as np
import matplotlib.pyplot as plt
from skspatial.objects import Point, Line, Vector


nameModel = 'input2/out_scaled.osim'
nameIK    = 'input2/out_ik.mot'
nameGRF   = 'input2/exp_grf.mot'


model = osim.Model(nameModel)
state = model.initSystem()
ground = model.getGround()

# read kinematics file
q = osim.TimeSeriesTable(nameIK)
t = list(q.getIndependentColumn())
dt = t[1]-t[0]
osim.TableUtilities().filterLowpass(q, 15, padData=True)
model.getSimbodyEngine().convertDegreesToRadians(q)
q.trim(t[0], t[-1])

f = osim.Storage(nameGRF) # read GRF file
f.resampleLinear(dt) # resample and trim the force file to match the IK file
f.crop(t[0],t[-1])
# Storage::exportTable() DOESN'T WORK IN PYTHON
# f.printToFile(f'{nameGRF[:-4]}_resampled.mot', dt)
f.printToXML(f'{nameGRF[:-4]}_resampled.mot')

f = osim.TimeSeriesTable(f'{nameGRF[:-4]}_resampled.mot')
f.getColumnLabels()

foot = dict()
for side,s in zip(['right','left'],['r','l']):
	foot[side] = dict()
	for i,ii in zip(['COP','GRV'],['p','v']):
		foot[side][i] = np.vstack([ f.getDependentColumn(f'ground_force_{s}_{ii}x').to_numpy(), \
									f.getDependentColumn(f'ground_force_{s}_{ii}y').to_numpy(), \
									f.getDependentColumn(f'ground_force_{s}_{ii}z').to_numpy()]).T
	foot[side]['MAG'] = np.linalg.norm(foot[side]['GRV'], ord=2, axis=1) > 10
	foot[side]['MA']  = np.empty((len(t),3))

# ground_force_r_vx
# ground_force_r_vy
# ground_force_r_vz

COM = np.empty((len(t),3)) # center of mass position

for i,ii in enumerate(t):

	##### Update coordinates' values and speeds
	value = osim.RowVector(q.getRowAtIndex(i))
	for j,coordinate in enumerate(model.updCoordinateSet()):
		coordinate.setValue(state, value[j], False)

	model.assemble(state)
	model.realizePosition(state)

	COM[i,:] = model.calcMassCenterPosition(state).to_numpy()


# https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
# %%
def momentArm(ax, plot=False, c='tab:blue'):
	line = Line(point=cop, direction=Vector(grv)/1000) # 
	# proj = line.project_point(com).to_array()
	dist = -1 * line.side_point(com) * line.distance_point(com)

	if plot:
		line.plot_2d(ax, alpha=0.1, c=c)
		Point(cop).plot_2d( ax, c='k', s=2)
		# Point(com).plot_2d( ax, c='g', s=2)
		# Point(proj).plot_2d(ax, c='r', s=2)

	return dist



plt.close('all')
# [6.4, 4.8] default figsize
_, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3,3, \
		figsize=(9.6,7.4), tight_layout=True, gridspec_kw={'height_ratios': [1,0.6,0.6]})
plt.suptitle('GRF Moment Arm and Moment about Center of Mass', fontsize=14, fontweight='bold')

for i,ii in enumerate(t):

	Point(COM[i,[0,1]]).plot_2d(ax1, c='g', s=2) # sagittal
	Point(COM[i,[2,1]]).plot_2d(ax2, c='g', s=2) # frontal
	Point(COM[i,[2,0]]).plot_2d(ax3, c='g', s=2) # transverse

	for side,color in zip(['right','left'],['tab:blue','tab:orange']):

		if foot[side]['MAG'][i]: # force data > 0

			# sagittal plane [XY]
			# [i,:2]
			com = COM[i,[0,1]]
			cop = foot[side]['COP'][i,[0,1]]
			grv = foot[side]['GRV'][i,[0,1]]
			foot[side]['MA'][i,0]= momentArm(ax1, plot=True, c=color)

			# frontal plane [ZY]
			# [i,::-1][:2] reverse the list
			com = COM[i,[2,1]]
			cop = foot[side]['COP'][i,[2,1]]
			grv = foot[side]['GRV'][i,[2,1]]
			foot[side]['MA'][i,1] = momentArm(ax2, plot=True, c=color)

			# transverse plane [ZX]
			com = COM[i,[2,0]]
			cop = foot[side]['COP'][i,[2,0]]
			grv = foot[side]['GRV'][i,[2,0]]
			foot[side]['MA'][i,2] = momentArm(ax3, plot=True, c=color)
		
		else:
			foot[side]['MA'][i,:] = np.ones(3) * np.nan




ax1.set_title('Sagittal Plane', fontweight='bold')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax2.set_title('Frontal Plane', fontweight='bold')
ax2.set_xlabel('Z (m)')
ax2.set_ylabel('Y (m)')
ax3.set_title('Transverse Plane', fontweight='bold')
ax3.set_xlabel('Z (m)')
ax3.set_ylabel('X (m)')


for side,color in zip(['right','left'],['tab:blue','tab:orange']):
	# moment arm
	ax4.plot(t, foot[side]['MA'][:,0], c=color) # sagittal
	ax5.plot(t, foot[side]['MA'][:,1], c=color) # frontal
	ax6.plot(t, foot[side]['MA'][:,2], c=color) # transverse
	# moment
	ax7.plot(t, foot[side]['MA'][:,0]*np.linalg.norm(foot[side]['GRV'][:,[0,1]], ord=2, axis=1), c=color)
	ax8.plot(t, foot[side]['MA'][:,1]*np.linalg.norm(foot[side]['GRV'][:,[2,1]], ord=2, axis=1), c=color)
	ax9.plot(t, foot[side]['MA'][:,2]*np.linalg.norm(foot[side]['GRV'][:,[2,0]], ord=2, axis=1), c=color)




for i in [ax4,ax5,ax6]:
	i.set_xlabel('Time (s)')
	i.set_ylabel('Moment Arm (m)')
	i.axhline(0, c='k', ls='--', alpha=0.25)
for i in [ax7,ax8,ax9]:
	i.set_xlabel('Time (s)')
	i.set_ylabel('Moment (Nm)')
	i.axhline(0, c='k', ls='--', alpha=0.25)




ax2.legend(['COM'])
ax5.legend(['right', 'left'])
plt.savefig('image.png', dpi=300)
# plt.show(block=False)


