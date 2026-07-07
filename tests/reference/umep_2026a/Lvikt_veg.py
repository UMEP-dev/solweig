# VENDORED UPSTREAM REFERENCE — do not import from library code.
#
# Source:  https://github.com/UMEP-dev/UMEP-processing
# Path:    functions/SOLWEIGpython/Lvikt_veg.py
# Copied from the pip `umep` package (0.0.1a18) — identical upstream (post commits 6c1f868 "Reimplement Lside
#          from v2015a" 2026-05-27 and 4c60f7f "Change name Lside_veg"
#          2026-05-29; ground_surface.py last touched df6f2d3 2026-06-04)
# License: GPL-3.0 (same as this repository and upstream UMEP)
#
# Kept verbatim (below this comment block) as the parity reference for the
# 2026a ground-surface-scheme port (see tests/spec/test_parity_2026a.py).
# The pip `umep` package (our other parity reference) ships only the 2025a
# generation; delete this copy and switch the tests to the pip package once
# upstream publishes a release containing the 2026a code.

def Lvikt_veg(svf,svfveg,svfaveg,vikttot):

    # Least
    viktonlywall=(vikttot-(63.227*svf**6-161.51*svf**5+156.91*svf**4-70.424*svf**3+16.773*svf**2-0.4863*svf))/vikttot

    viktaveg=(vikttot-(63.227*svfaveg**6-161.51*svfaveg**5+156.91*svfaveg**4-70.424*svfaveg**3+16.773*svfaveg**2-0.4863*svfaveg))/vikttot

    viktwall=viktonlywall-viktaveg

    svfvegbu=(svfveg+svf-1)  # Vegetation plus buildings
    viktsky=(63.227*svfvegbu**6-161.51*svfvegbu**5+156.91*svfvegbu**4-70.424*svfvegbu**3+16.773*svfvegbu**2-0.4863*svfvegbu)/vikttot
    viktrefl=(vikttot-(63.227*svfvegbu**6-161.51*svfvegbu**5+156.91*svfvegbu**4-70.424*svfvegbu**3+16.773*svfvegbu**2-0.4863*svfvegbu))/vikttot
    viktveg=(vikttot-(63.227*svfvegbu**6-161.51*svfvegbu**5+156.91*svfvegbu**4-70.424*svfvegbu**3+16.773*svfvegbu**2-0.4863*svfvegbu))/vikttot
    viktveg=viktveg-viktwall

    return viktveg,viktwall,viktsky,viktrefl