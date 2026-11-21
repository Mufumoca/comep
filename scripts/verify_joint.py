import numpy as np

from comep.joint import fit_joint, probabilities


def main():
    cells=np.array([[1,0,0,0],[1,1,0,0],[1,0,1,0],[1,1,1,1]],dtype=float)
    schedule=np.tile(np.arange(4),1000)
    design=cells[schedule]
    rewards=(np.random.default_rng(1234).random(len(schedule))<probabilities(design@[-1,1,1,-3])).astype(int)
    fit=fit_joint(design,rewards)
    assert fit.converged and fit.rank==4
    assert fit.contrast([0,0,0,1])["status"]=="negative"
    scaled=fit_joint(design*[1,.25,.25,.0625],rewards)
    np.testing.assert_allclose(fit.predict(cells),scaled.predict(cells*[1,.25,.25,.0625]),atol=1e-10)
    np.testing.assert_allclose(fit.contrast([0,0,0,1])["mean"],scaled.contrast([0,0,0,.0625])["mean"])
    np.testing.assert_allclose(fit.contrast([0,0,0,1])["approximate_standard_error"],scaled.contrast([0,0,0,.0625])["approximate_standard_error"],atol=1e-10)
    assert fit.contrast([0,0,0,1])["identifiable"] == scaled.contrast([0,0,0,.0625])["identifiable"]
    normalized=design/fit.scales
    probability=fit.predict(design)
    information=normalized.T@((probability*(1-probability))[:,None]*normalized)+.1*np.eye(4)
    np.testing.assert_allclose(fit.covariance@information,np.eye(4),atol=1e-9)
    confounded=fit_joint(np.ones((500,4)),np.zeros(500))
    assert confounded.rank==1 and confounded.contrast([0,0,0,1])["status"]=="unknown"
    assert confounded.contrast([1,1,1,1])["identifiable"]
    unseen=fit_joint(np.array([[1,0],[1,0]]),[0,1])
    assert unseen.contrast([0,1])["status"]=="unknown"
    for design,rewards in [(np.ones((3,2)),[0,1]),(np.ones((2,2)),[0,2]),(np.array([[1,np.nan]]),[0])]:
        try:
            fit_joint(design,rewards)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid joint inputs accepted")
    print("JOINT_WITNESS scaled predictions invariant; rank1 unknown; fitted contrast negative")


if __name__=="__main__":
    main()
