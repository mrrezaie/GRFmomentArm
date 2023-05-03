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

# read GRF file
f = osim.Storage(nameGRF)
# resample and trim the force file to match the IK file
f.resampleLinear(dt)
f.crop(t[0],t[-1])
# Storage::exportTable() DOESN'T WORK IN PYTHON
# f.printToFile(f'{nameGRF[:-4]}_resampled.mot', dt)
f.printToXML(f'{nameGRF[:-4]}_resampled.mot')

f = osim.TimeSeriesTable(f'{nameGRF[:-4]}_resampled.mot')
# f.trim(t[0],t[-1])
f.getColumnLabels()
# f.getNumColumns()/9
COP = np.vstack([f.getDependentColumn('ground_force_r_px').to_numpy(), \
				f.getDependentColumn('ground_force_r_py').to_numpy(), \
				f.getDependentColumn('ground_force_r_pz').to_numpy()]).T
GRV = np.vstack([f.getDependentColumn('ground_force_r_vx').to_numpy(), \
				f.getDependentColumn('ground_force_r_vy').to_numpy(), \
				f.getDependentColumn('ground_force_r_vz').to_numpy()]).T
# time of foot contact (force data exist)
magnitude = np.linalg.norm(GRV, ord=2, axis=1) > 0
# ok = np.where(magnitude)[0].tolist()

# COP = COP[ok,:]
# GRV = GRV[ok,:]

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

def momentArm(ax, plot=False):
	line = Line(point=cop, direction=Vector(grv)/1000) # 
	proj = line.project_point(com).to_array()
	dist = -1 * line.side_point(com) * line.distance_point(com)
	if plot:
		line.plot_2d(ax, alpha=0.1, c='tab:blue')
		Point(cop).plot_2d( ax, c='k', s=2)
		Point(com).plot_2d( ax, c='g', s=2)
		Point(proj).plot_2d(ax, c='r', s=2)
	return dist


plt.close('all')
# [6.4, 4.8] default figsize
_, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3,3, \
		figsize=(9.6,7.4), tight_layout=True, gridspec_kw={'height_ratios': [1,0.6,0.6]})
plt.suptitle('GRF Moment Arm and Moment about Center of Mass', fontsize=14, fontweight='bold')

dist = np.empty((len(t),3))
for i,ii in enumerate(t):
	if magnitude[i] > 0:
		# sagittal plane [XY]
		# [i,:2]
		com = COM[i,[0,1]]
		cop = COP[i,[0,1]]
		grv = GRV[i,[0,1]]
		dist[i,0] = momentArm(ax1, plot=True)

		# frontal plane [ZY]
		# [i,::-1][:2] reverse the list
		com = COM[i,[2,1]]
		cop = COP[i,[2,1]]
		grv = GRV[i,[2,1]]
		dist[i,1] = momentArm(ax2, plot=True)

		# transverse plane [ZX]
		com = COM[i,[2,0]]
		cop = COP[i,[2,0]]
		grv = GRV[i,[2,0]]
		dist[i,2] = momentArm(ax3, plot=True)
	
	else:
		dist[i,:] = np.ones(3) * np.nan

ax1.set_title('Sagittal Plane', fontweight='bold')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax2.set_title('Frontal Plane', fontweight='bold')
ax2.set_xlabel('Z (m)')
ax2.set_ylabel('Y (m)')
ax3.set_title('Transverse Plane', fontweight='bold')
ax3.set_xlabel('Z (m)')
ax3.set_ylabel('X (m)')
for i in [ax4,ax5,ax6]:
	i.set_xlabel('Time (s)')
	i.set_ylabel('Moment Arm (m)')
	i.axhline(0, c='k', ls='--', alpha=0.25)
for i in [ax7,ax8,ax9]:
	i.set_xlabel('Time (s)')
	i.set_ylabel('Moment (Nm)')
	i.axhline(0, c='k', ls='--', alpha=0.25)
ax4.plot(t, dist[:,0])
ax5.plot(t, dist[:,1])
ax6.plot(t, dist[:,2])
ax7.plot(t, dist[:,0]*np.linalg.norm(GRV[:,[0,1]], ord=2, axis=1))
ax8.plot(t, dist[:,1]*np.linalg.norm(GRV[:,[2,1]], ord=2, axis=1))
ax9.plot(t, dist[:,2]*np.linalg.norm(GRV[:,[2,0]], ord=2, axis=1))
ax2.legend(['unit GRV', 'COP', 'COM', 'COMproj'])
plt.savefig('image.png', dpi=300)
# plt.show(block=False)


