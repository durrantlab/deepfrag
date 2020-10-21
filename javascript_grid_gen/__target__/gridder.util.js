'use strict';var math={};import{AssertionError,AttributeError,BaseException,DeprecationWarning,Exception,IndexError,IterableError,KeyError,NotImplementedError,RuntimeWarning,StopIteration,UserWarning,ValueError,Warning,__JsIterator__,__PyIterator__,__Terminal__,__add__,__and__,__call__,__class__,__envir__,__eq__,__floordiv__,__ge__,__get__,__getcm__,__getitem__,__getslice__,__getsm__,__gt__,__i__,__iadd__,__iand__,__idiv__,__ijsmod__,__ilshift__,__imatmul__,__imod__,__imul__,__in__,__init__,__ior__,
__ipow__,__irshift__,__isub__,__ixor__,__jsUsePyNext__,__jsmod__,__k__,__kwargtrans__,__le__,__lshift__,__lt__,__matmul__,__mergefields__,__mergekwargtrans__,__mod__,__mul__,__ne__,__neg__,__nest__,__or__,__pow__,__pragma__,__proxy__,__pyUseJsNext__,__rshift__,__setitem__,__setproperty__,__setslice__,__sort__,__specialattrib__,__sub__,__super__,__t__,__terminal__,__truediv__,__withblock__,__xor__,abs,all,any,assert,bool,bytearray,bytes,callable,chr,copy,deepcopy,delattr,dict,dir,divmod,enumerate,
filter,float,getattr,hasattr,input,int,isinstance,issubclass,len,list,map,max,min,object,ord,pow,print,property,py_TypeError,py_iter,py_metatype,py_next,py_reversed,py_typeof,range,repr,round,set,setattr,sorted,str,sum,tuple,zip}from"./org.transcrypt.__runtime__.js";import*as __module_math__ from"./math.js";__nest__(math,"",__module_math__);import{Chem}from"./gridder.fake_rdkit.js";var __name__="gridder.util";export var get_coords=function(mol){var conf=mol.GetConformer();var coords=function(){var __accu0__=
[];for(var i=0;i<conf.GetNumAtoms();i++)__accu0__.append(conf.GetAtomPosition(i));return __accu0__}();return coords};export var get_atomic_nums=function(mol){return function(){var __accu0__=[];for(var i=0;i<mol.GetNumAtoms();i++)__accu0__.append(mol.GetAtomWithIdx(i).GetAtomicNum());return __accu0__}()};export var generate_fragments=function(mol,max_heavy_atoms,only_single_bonds){var max_heavy_atoms=max_heavy_atoms===null?0:max_heavy_atoms;var only_single_bonds=only_single_bonds===null?true:only_single_bonds;
var splits=[];var splits=[tuple([mol,null])];return splits};export var load_receptor=function(rec_path){var rec=Chem.MolFromPDBFile(rec_path,__kwargtrans__({sanitize:false}));var rec=remove_water(rec);var rec=remove_hydrogens(rec);return rec};export var remove_hydrogens=function(m){m.atoms=function(){var __accu0__=[];for(var a of m.atoms)if(a.element!="H")__accu0__.append(a);return __accu0__}();return m};export var remove_water=function(m){m.atoms=function(){var __accu0__=[];for(var a of m.atoms)if(!__in__(a.resname,
["WAT","HOH","TIP","TIP3","OH2"]))__accu0__.append(a);return __accu0__}();var merged=m;return merged};export var combine_all=function(frags){if(len(frags)==0)return null;var c=frags[0];for(var f of frags.__getslice__(1,null,1))var c=Chem.CombineMols(c,f);return c};export var load_ligand=function(sdf){var lig=py_next(Chem.SDMolSupplier(sdf,__kwargtrans__({sanitize:false})));var lig=remove_water(lig);var lig=remove_hydrogens(lig);var frags=generate_fragments(lig,null,null);return tuple([lig,frags])};
export var mol_to_points=function(mol,atom_types,note_sulfur){if(typeof note_sulfur=="undefined"||note_sulfur!=null&&note_sulfur.hasOwnProperty("__kwargtrans__"))var note_sulfur=true;var atom_types=atom_types===null?[6,7,8,16]:atom_types;var coords=get_coords(mol);var atomic_nums=get_atomic_nums(mol);var layers=[];for(var t of atomic_nums)if(t==1)layers.append(-1);else if(t==6)layers.append(0);else if(t==7)layers.append(1);else if(t==8)layers.append(2);else if(!note_sulfur)layers.append(3);else if(t==
16)layers.append(3);else layers.append(4);var coords=function(){var __accu0__=[];for(var [i,c]of enumerate(coords))if(layers[i]!=-1)__accu0__.append(c);return __accu0__}();var layers=function(){var __accu0__=[];for(var l of layers)if(l!=-1)__accu0__.append(l);return __accu0__}();return tuple([coords,layers])};export var get_connection_point=function(frag){var dummy_idx=get_atomic_nums(frag).index(0);var coords=get_coords(frag)[dummy_idx];return coords};export var frag_dist_to_receptor_raw=function(coords,
frag){var conn=get_connection_point(frag);var dists=[];for(var i=0;i<len(coords);i++){var coord=coords[i];var tmp=[coord[0]-conn.x,coord[1]-conn.y,coord[2]-conn.z];var tmp=[Math.pow(tmp[0],2),Math.pow(tmp[1],2),Math.pow(tmp[2],2)];var s=sum(tmp);dists.append(s)}var min_dist=math.sqrt(min(dists));return min_dist};export var mol_array=function(mol){var coords=get_coords(mol);var types=get_atomic_nums(mol);var arr=[];for(var [i,coor]of enumerate(coords))arr.append([coor.x,coor.y,coor.z,types[i]]);return arr};

//# sourceMappingURL=gridder.util.map