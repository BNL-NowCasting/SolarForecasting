import argparse
import copy
from datetime import datetime
import glob
import os
import uuid
import yaml

# TODO:
#	reserved words
#		(unnamed, metadata)
#	readability

# Configuration file to use if a value isn't passed
CONF_F = "/home/tchapman/root/configs/config.yaml"

# do string concatenation so I can use variables in the YAML files
# this is automatically invoked by yaml.safe_load
class Join( yaml.YAMLObject ):
    yaml_loader = yaml.SafeLoader
    yaml_tag = "!join"
    
    @classmethod
    def from_yaml(cls, loader, node):
        return "".join([s.value for s in node.value])

### Utility methods

# combine the dictionaries d1 and d2 with
# values from d2 replacing those in d1 when keys overlap
# modifies d1 in place
# loops forever if dictionary has a loop
def recursive_merge( d1, d2 ):
    for key, val in d2.items():
        if isinstance( val, dict ):
            d1.setdefault( key, {} )
            # if not merged( d1[key] ):
            recursive_merge( d1[key], val )
        else:
            d1[key] = val
    return d1

# remove all entries in d1 which exist in d2
# loops forever if dictionary has a loop
def diff_config( d1, d2 ):
    out_dict = {}
    for key, val in d1.items():
        if not key in d2:
            out_dict[key] = val
        else:
            if isinstance( val, dict ):
                # if not diffed( d2[key] ):
                tmp_dict = diff_config( val, d2[key] )
                if len( list( tmp_dict.keys() ) ):
                    out_dict[key] = tmp_dict
            elif d1[key] != d2[key]:
                out_dict[key] = val
                
    return out_dict

# load the given configuration file and
# cast all the top level headers to string so I can compare them to strings
def load_config(f):
    config = yaml.safe_load(f)
    out_config = {}
    for key, value in config.items():
        out_config[str(key)] = value
    return out_config

def find_and_load_config( conf_name, main_config_f ):
    return find_and_load_configs( [conf_name], main_config_f )

# conf_name: key of the config to load
# main_config_f: configuration file containing default header
def find_and_load_configs( conf_names, main_config_f ):
    configs = {}
    with open( main_config_f, "r" ) as f:
        main_config = load_config(f)
        default = main_config["default"]
        for conf_name in conf_names:
            if conf_name in main_config:
                configs[conf_name] = main_config[conf_name]

    # if some conf_names weren't in main_config_f...
    folder = os.path.dirname( main_config_f )
    files = glob.glob( os.path.join( folder, "generated_configs_*.yaml" ) )
    for fn in files:
        if len(configs.keys()) == len(conf_names):
            break
        with open( fn, "r" ) as f:
            generated_config = load_config(f)
            for conf_name in conf_names:
                if conf_name in generated_config:
                    configs[conf_name] = generated_config[conf_name]

    if len(configs.keys()) < len(conf_names):
        diff = list( set(conf_names) - set(configs.keys()) )
        conf_names = list(configs.keys())
        #logger.log( "Error: failed to find configs " + str(diff) )
        print( "Error: failed to find configs " + str(diff) )

    for name in conf_names:
        # modifies conf in-place
        recursive_merge( default, configs[name] ) 
    return default

# recursively generate yaml specification of dict config
def dump_config( config, indent=0, recurse=True ):
    text = ""

    keys = list(config.keys())
    keys.sort()
    for k in keys:
        v = config[k]
        if isinstance( v, dict ):
            text += " "*indent + "{}: \n".format(k)
            if recurse:
                # if printed( k ):
                    # text += " "*indent + "[loop]\n"
                # else:
                text += dump_config( v, indent=indent+2 )
        else:
            text += " "*indent + "{}: {}\n".format( k, v )
    return text
    

def save_config( config, config_fn, name="", metadata={} ):
    key = str(uuid.uuid1().int)

    generated = False
    if not name:
        name = key
        generated = True

    # store metadata dictionary with the new configuration
    config["metadata"] = {
        "timestamp": datetime.now(),
         "key": key,
        "name": name
    }
    # add any metadata provided by the invoking script
    recursive_merge( config["metadata"], metadata )

    save_file = config_fn
    if generated:
        folder = os.path.dirname( config_fn )
        year = datetime.now().year
        save_file = os.path.join( folder, "generated_configs_{}.yaml".format( year ) )

    if not os.path.exists( save_file ):
        os.mknod( save_file )

    with open( save_file, "r+" ) as f:
        skipping = False

        d = f.readlines()
        f.seek(0)
        for l in d:
            # erase the previous version of this config if any
            if l.find(name+":") == 0:
                skipping = True
            elif l[0] != " ":
                skipping = False

            if not skipping:
                f.write(l)

        text = dump_config( {name: config} )
        f.write( text )
        f.truncate()
    return name

def build_parser():
    parser = argparse.ArgumentParser(description="Handle config selection and creation")
    parser.add_argument( 'source', nargs="?", default=CONF_F, 
                         help="The configuration file to read from" )
    parser.add_argument( '-c', '--config', action='append', default=[], 
                         help="Load an existing configuration with key CONFIG. Can be used multiple times; if multiple loaded configs share attributes, the last one given takes precendence." )
    parser.add_argument( '-n', '--new', nargs='?', const="unnamed", 
       help="Create a new configuration file with name NAME (defaults to a random key). Can be used in conjuction with -c to change the initial state of your new configuration." )
    parser.add_argument( '-e', '--edit', default="",
                         help="Edit an existing config file. When used in conjunction with -c, loads the target file and then loads any configs from -c on top of it." )
    parser.add_argument( '-i', '--siteid', 
                         help="The site to use the configuration for." )
    parser.add_argument( '-S', '--start', action='append', default=[], 
                         help="Shortcut to create a new configuration using the given date ranges for pipeline/start_dates. Can be used multiple times. Cannot be used in conjuction with -n or -e" )
    parser.add_argument( '-E', '--end', action='append', default=[],
                         help="Shortcut to create a new configuration using the given date ranges for pipeline/end_dates. Can be used multiple times. Cannot be used in conjuction with -n or -e" )
    parser.add_argument( '-r', '--reprocess', action='append', default=[], 
                         const='all', nargs='?',
                         help="Redo the given preprocessing step. 'all' if no arg given. Can be used multiple times. Cannot be used in conjuction with -n or -e. Valid options: 'all', 'features', 'stitch', 'motion', 'height', 'image_preprocessing'" )
    parser.add_argument('-l', '--leadminutes', action='append', default=[],
                         help="Shortcut to create a new configuration using the given values for pipeline/lead_minutes. Can be used multiple times. Cannot be used in conjuction with -n or -e" )
    parser.add_argument('-s', '--sensors', action='append', default=[],
                         help="Shortcut to create a new configuration using the given values for pipeline/sensors. Can be used multiple times. Cannot be used in conjuction with -n or -e" )
    parser.add_argument('-u', '--update', nargs=2, action='append', default=[],
                         help="-u Shortcut to create set {var} to {value} under a header given by the invoking script. Can be used multiple times. Cannot be used in conjuction with -n or -e" )
    parser.add_argument('-p', '--path', nargs=2, action='append', default=[],
                         help="Shortcut to create {path_name} to {value}. Can be used multiple times. Cannot be used in conjuction with -n or -e" )
    return parser

### Public facing method
# process given arguments and
# construct and return a configuration accordingly
def handle_config(default_config_name="", metadata={}, parser=None, header=""):
    if header:
        invoking_header = header
    elif metadata and "invoking_script" in metadata:
        invoking_header = metadata["invoking_script"]

    if not parser:
        parser = build_parser()
    args = parser.parse_args()

    config_f = args.source 
    with open( config_f, "r" ) as f:
        all_configs = load_config(f)

    loaded_conf_names = args.config # non-default config files to use

    # shortcuts to run custom pipeline...
    if args.update or args.leadminutes or args.reprocess or args.start or args.end or args.sensors or args.path or args.siteid:
        if len(args.start) != len(args.end):
            print( "Number of --start and --end flags must match {} != {}".format( len(args.start), len(args.end) ) )
            exit(1)
        # process quick-config args and generate an unnamed configuration to use
        lead_minutes = args.leadminutes
        reprocess = args.reprocess
        target_start_dates = args.start
        target_end_dates = args.end
        explicit_updates = args.update
        path_updates = args.path
        site_id = args.siteid
    
        # build a new full_config using the loaded files
        conf = find_and_load_configs( loaded_conf_names, config_f )
    
        # update that full_config using the given quick-config arguments
        if lead_minutes:
            conf["pipeline"]["lead_minutes"] = lead_minutes
        if reprocess:
            for key in reprocess:
                if key == "all":
                    conf["pipeline"]["reprocess_all"] = True
                elif key in conf["pipeline"]:
                    conf["pipeline"][key]["reprocess"] = True
        if target_start_dates:
            conf["pipeline"]["target_ranges"]["start_dates"] = target_start_dates
            conf["pipeline"]["target_ranges"]["end_dates"] = target_end_dates
        if site_id:
            conf["site_id"] = site_id
        for p in path_updates:
            conf["paths"][p[0]] = p[1]
        for u in explicit_updates:
            if not invoking_header:
                print( "Error: no header given when config was initialized. Cannot process -u flags." )
                exit(1)
            conf[invoking_header][u[0]] = u[1]
    
        # remove any attributes that match default if save_full_configs is off
        default = {} if all_configs["meta"]["save_full_configs"] else all_configs["default"]
        diff = diff_config( conf, default )
        conf_name = save_config( diff, config_f, name="", metadata=metadata )
        
        # reload the config to deal with some issues with how data
        # entered in the editor is interpreted (e.g arrays are strings)
        return find_and_load_config( conf_name, config_f )
    else:
        # just load the given configurations and return their union
        conf = find_and_load_configs( loaded_conf_names, config_f )

        # for logging purposes:
        if not "metadata" in conf:
            conf["metadata"] = {}
        conf["metadata"]["name"] = loaded_conf_names # for logging
        return conf
