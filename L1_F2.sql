SELECT
p.subfield,  
p.alpha, 
p.delta, 
p.r, 
s.e1, 
s.e2, 
s.de, 
s.a, 
s.b, 
p.processflags, 
z.z_b, 
s.flux_radius, 
d.Dlsqc_prob
FROM
RC1Stage.PhotoObjAll AS p, 
RC1c_public.Dlsqc AS d, 
RC1c_public.Bpz AS z, 
RC1Stage.Shapes2 AS s 
WHERE
d.objid=s.objid
AND p.objid = s.objid
AND p.objid = z.objid
AND p.objid IS NOT NULL
AND p.processflags<8
AND p.r is NOT NULL
AND p.b is NOT NULL
AND p.v is NOT NULL
AND p.z is NOT NULL
AND p.r>18 AND p.r<21
# The R band probability that object is a point source `d.Dlsqc_prob`
AND d.Dlsqc_prob<0.1
# Shape cut
AND s.b>0.4
AND z.z_b>0.1
AND z.z_b<.4
# Ellipticity error cut
AND s.de<0.25
# F5 bound cut F5 is a 2x2 sq. degree field centered at RA=13:59:20, DEC=-11:03:00
# Due to ambiguous info on DLS website and James 2015 cosmic shear paper about the location
# of the field I will use the SQL keyword instead
# AND p.alpha between 208.7 and 210.85
# AND p.delta between -12.1 and -10.1;
AND p.subfield LIKE 'F2%'
